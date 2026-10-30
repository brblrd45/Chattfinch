"""The Telegram bot: message it a question, get an answer with its sources.

Run from the repo root:

    .venv/bin/python -m chattfinch.bot

Needs the built index, ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN and
TELEGRAM_ALLOWED_USER_IDS, set as environment variables (or in a local .env
file, which is never committed).

The bot uses long polling: it fetches messages over an outbound connection, so the
machine needs no open port. Only one process may poll a given bot token at a time.

Access is closed by default. Only user IDs on the allowlist reach the model; anyone
else is told the bot is private (and their own ID, which is how you find yours).
Logs record user IDs and timings, never message text.
"""

import asyncio
import hashlib
import logging
import sys
import time
import unicodedata
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes,
                          MessageHandler, filters)

from chattfinch import config, llm, rag, retrieval, router
from chattfinch import telegram_format as fmt

log = logging.getLogger("chattfinch.bot")

EXAMPLES = [
    "What does the Saker Falcon eat?",
    "Where do Demoiselle Cranes spend the winter?",
    "Why has the Sociable Lapwing declined?",
]
NO_PREVIEW = LinkPreviewOptions(is_disabled=True)

WELCOME = (f"<b>Ask a Birder</b>\n\nAsk me about the birds recorded in {config.REGION_NAME}. "
           "I answer only from Wikipedia and eBird, and show the sources so you can check me.\n\n"
           "Send a question, or try one of these:")
PRIVATE = "This bot is private. Your Telegram user ID is {user_id}."
TEXT_ONLY = f"I can only read text. Send me a question about a bird recorded in {config.REGION_NAME}."
EMPTY = "Send me a question about a bird."
TOO_LONG = "Please keep your question under {limit} characters."
RATE_LIMITED = "You've asked a lot of questions in a short time. Please try again later."
FAILED = "Sorry, I couldn't produce an answer just now. Please try again in a moment."
EXPIRED = "That answer has expired. Ask the question again to see its passages."
CHOICE_EXPIRED = "That question has expired. Please ask it again."
SHOW_PASSAGES = "Show passages"


def parse_user_ids(value: str) -> frozenset[int]:
    """Numeric IDs from a comma-separated string. Raises ValueError on anything else."""
    ids = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"TELEGRAM_ALLOWED_USER_IDS must be numeric user IDs separated by "
                             f"commas; got {part!r}")
        ids.add(int(part))
    return frozenset(ids)


def normalize_question(question: str) -> str:
    """The question with case, punctuation and spacing ignored, so "What does it
    eat?" and "what does it eat" are one question. Letters in any script are kept:
    the bird-name normalization (aliases.normalize) drops non-Latin letters, which
    would make every question typed in Cyrillic the same question."""
    text = unicodedata.normalize("NFKC", question).casefold()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())


def question_key(question: str, choice: str | None = None) -> str:
    """A short stable key for a question (and the bird chosen for it, if one had
    to be), as normalize_question sees it. Used as the cache key and inside button
    data (Telegram allows 64 bytes there)."""
    normalized = normalize_question(question)
    if choice is not None:
        normalized += "\n" + choice
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


class RateLimiter:
    """At most `max_calls` per `period_s` for each user (a sliding window)."""

    def __init__(self, max_calls: int, period_s: float, clock: Callable[[], float] = time.monotonic):
        self.max_calls, self.period_s, self._clock = max_calls, period_s, clock
        self._calls: dict[int, deque[float]] = defaultdict(deque)

    def allow(self, user_id: int) -> bool:
        """True, and the call is counted, unless the user is over the limit."""
        now, calls = self._clock(), self._calls[user_id]
        while calls and now - calls[0] >= self.period_s:
            calls.popleft()
        if len(calls) >= self.max_calls:
            return False
        calls.append(now)
        return True


# Answers that are never cached: a failure to reach eBird or to find the IUCN
# snapshot is temporary, and the next ask should try again.
UNCACHED_REASONS = {router.LIVE_UNAVAILABLE, router.STATUS_UNAVAILABLE}


class AnswerCache:
    """The most recent answers by question key, so a repeated question doesn't pay
    for the model again and "Show passages" can find the answer it belongs to.

    Route-aware: a live ("seen recently?") answer is kept only `live_ttl_s`, long
    enough for its "Show passages" button, short enough that a later ask fetches
    fresh sightings. Temporary failures aren't kept at all. Everything else stays
    until it is the least recently used of `max_entries`."""

    def __init__(self, max_entries: int = 200, live_ttl_s: float = config.LIVE_CACHE_SECONDS,
                 clock: Callable[[], float] = time.monotonic):
        self.max_entries, self.live_ttl_s, self._clock = max_entries, live_ttl_s, clock
        self._answers: OrderedDict[str, tuple[rag.Answer, float | None]] = OrderedDict()

    def get(self, key: str) -> rag.Answer | None:
        if key not in self._answers:
            return None
        answer, expires = self._answers[key]
        if expires is not None and self._clock() >= expires:
            del self._answers[key]
            return None
        self._answers.move_to_end(key)
        return answer

    def put(self, key: str, answer: rag.Answer) -> None:
        if answer.decline_reason in UNCACHED_REASONS:
            self._answers.pop(key, None)
            return
        expires = self._clock() + self.live_ttl_s if answer.route == router.LIVE else None
        self._answers[key] = (answer, expires)
        self._answers.move_to_end(key)
        while len(self._answers) > self.max_entries:
            self._answers.popitem(last=False)


class Chattfinch:
    """The bot's behavior, independent of the Telegram connection: handlers take
    an update and reply through it. `answer_fn` is rag.answer unless a test passes
    a fake."""

    def __init__(self, allowed_user_ids: frozenset[int],
                 answer_fn: Callable[..., rag.Answer] = rag.answer,
                 rate_limiter: RateLimiter | None = None, cache: AnswerCache | None = None):
        self.allowed_user_ids = allowed_user_ids
        self.answer_fn = answer_fn
        self.rate_limiter = rate_limiter or RateLimiter(config.TELEGRAM_RATE_LIMIT_QUESTIONS,
                                                        config.TELEGRAM_RATE_LIMIT_SECONDS)
        self.cache = cache or AnswerCache()
        # One answer at a time: the embedding model and the index aren't worth
        # sharing across threads, and it bounds how fast the API key can be spent.
        self._pipeline = asyncio.Lock()

    # --- replies ---------------------------------------------------------------------

    async def _send(self, update: Update, text_html: str,
                    markup: InlineKeyboardMarkup | None = None) -> None:
        """Reply with HTML; if Telegram rejects the markup, resend as plain text
        rather than leave the user without an answer."""
        message = update.effective_message
        try:
            await message.reply_text(text_html, parse_mode=ParseMode.HTML,
                                     link_preview_options=NO_PREVIEW, reply_markup=markup)
        except BadRequest as error:
            log.warning("Telegram rejected an HTML message (%s); resending as plain text", error)
            await message.reply_text(fmt.plain(text_html), link_preview_options=NO_PREVIEW,
                                     reply_markup=markup)

    async def _allowed(self, update: Update) -> bool:
        """True if the sender is on the allowlist; otherwise tells them the bot is
        private. Nothing past this check runs for anyone else."""
        user = update.effective_user
        if user is not None and user.id in self.allowed_user_ids:
            return True
        if update.callback_query is not None:
            await update.callback_query.answer()
        log.info("refused user=%s", getattr(user, "id", None))
        await self._send(update, PRIVATE.format(user_id=getattr(user, "id", "unknown")))
        return False

    # --- handlers --------------------------------------------------------------------

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        """/start, /help and unknown commands: what the bot does, with examples."""
        if not await self._allowed(update):
            return
        buttons = [[InlineKeyboardButton(example, callback_data=f"ex:{i}")]
                   for i, example in enumerate(EXAMPLES)]
        await self._send(update, WELCOME, InlineKeyboardMarkup(buttons))

    async def question(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        """A text message: answer it."""
        if await self._allowed(update):
            await self._answer(update, update.effective_message.text or "")

    async def unsupported(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        """Photos, voice notes, stickers and so on."""
        if await self._allowed(update):
            await self._send(update, TEXT_ONLY)

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        """A button press: an example question, a choice between birds, or "Show
        passages"."""
        if not await self._allowed(update):
            return
        query = update.callback_query
        await query.answer()  # stops the button's loading spinner
        kind, _, value = (query.data or "").partition(":")
        if kind == "ex" and value.isdigit() and int(value) < len(EXAMPLES):
            await self._answer(update, EXAMPLES[int(value)])
        elif kind == "c":  # c:<question key>:<candidate number>
            key, _, number = value.partition(":")
            asked = self.cache.get(key)
            if asked is None or not asked.needs_choice or not number.isdigit() \
                    or int(number) >= len(asked.choices.species):
                await self._send(update, CHOICE_EXPIRED)
                return
            try:  # one choice per question: remove the buttons
                await query.edit_message_reply_markup(reply_markup=None)
            except BadRequest:
                pass
            await self._answer(update, asked.query, choice=asked.choices.species[int(number)])
        elif kind == "p":
            answer = self.cache.get(value)
            if answer is None:
                await self._send(update, EXPIRED)
                return
            log.info("passages user=%s sources=%d", update.effective_user.id, len(answer.sources))
            for message in fmt.passage_messages(answer):
                await self._send(update, message)
            try:  # the passages are shown; remove the button so they aren't sent twice
                await query.edit_message_reply_markup(reply_markup=None)
            except BadRequest:
                pass

    # --- the answer flow -------------------------------------------------------------

    async def _answer(self, update: Update, question: str, choice: str | None = None) -> None:
        user_id = update.effective_user.id
        question = question.strip()
        if not question:
            await self._send(update, EMPTY)
            return
        if len(question) > config.TELEGRAM_MAX_QUESTION_CHARS:
            await self._send(update, TOO_LONG.format(limit=config.TELEGRAM_MAX_QUESTION_CHARS))
            return

        key = question_key(question, choice)
        answer = self.cache.get(key)
        cached = answer is not None
        started = time.monotonic()
        if not cached:
            if not self.rate_limiter.allow(user_id):
                log.info("rate-limited user=%s", user_id)
                await self._send(update, RATE_LIMITED)
                return
            await update.effective_message.reply_chat_action(ChatAction.TYPING)
            try:
                async with self._pipeline:
                    # The pipeline is synchronous and takes seconds; a worker thread
                    # keeps the bot responsive to other messages meanwhile.
                    if choice is None:
                        answer = await asyncio.to_thread(self.answer_fn, question)
                    else:
                        answer = await asyncio.to_thread(self.answer_fn, question,
                                                         choice=choice)
            except llm.LLMError as error:
                log.error("model error user=%s: %s", user_id, error)
                await self._send(update, FAILED)
                return
            except Exception:
                log.exception("answer failed user=%s", user_id)
                await self._send(update, FAILED)
                return
            self.cache.put(key, answer)

        if answer.needs_choice:
            log.info("asked to choose user=%s candidates=%d", user_id,
                     len(answer.choices.species))
            buttons = [[InlineKeyboardButton(name, callback_data=f"c:{key}:{number}")]
                       for number, name in enumerate(answer.choices.common_names)]
            await self._send(update, fmt.answer_messages(answer)[0], InlineKeyboardMarkup(buttons))
            return
        # Counts and fixed labels only, never text: the decline reason is one of
        # prompts.DECLINE_REASONS or router's, and names/renamed count the birds the question
        # named and how many of those it named by another name.
        log.info("answered user=%s route=%s declined=%s reason=%s names=%d renamed=%d chosen=%s "
                 "cached=%s sources=%d seconds=%.1f", user_id, answer.route, answer.declined,
                 answer.decline_reason or "-", len(answer.names), len(answer.renamed),
                 choice is not None, cached, len(answer.sources), time.monotonic() - started)
        messages = fmt.answer_messages(answer)
        markup = None
        # No passages button when the passages are unrelated to the question (the bird
        # wasn't found): showing them would suggest they have something to do with it.
        if answer.sources and answer.sources_are_relevant:
            markup = InlineKeyboardMarkup(
                [[InlineKeyboardButton(SHOW_PASSAGES, callback_data=f"p:{key}")]])
        for message in messages[:-1]:
            await self._send(update, message)
        await self._send(update, messages[-1], markup)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("unhandled error", exc_info=context.error)


def build_application(token: str, bot: Chattfinch) -> Application:
    # concurrent_updates: without it the library handles one message at a time, so
    # a slow answer would hold up everyone else's messages.
    application = Application.builder().token(token).concurrent_updates(True).build()
    private = filters.ChatType.PRIVATE  # private chats only; groups are ignored
    application.add_handler(CommandHandler(["start", "help"], bot.start, filters=private))
    application.add_handler(MessageHandler(private & filters.COMMAND, bot.start))
    application.add_handler(MessageHandler(private & filters.TEXT, bot.question))
    application.add_handler(MessageHandler(private, bot.unsupported))
    application.add_handler(CallbackQueryHandler(bot.button))
    application.add_error_handler(on_error)
    return application


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # The HTTP library logs every request URL at INFO, and Telegram URLs contain the
    # bot token. Keep it out of the logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if not config.TELEGRAM_BOT_TOKEN:
        sys.exit("TELEGRAM_BOT_TOKEN is not set. Create a bot with @BotFather and add its "
                 "token to .env.")
    try:
        allowed = parse_user_ids(config.TELEGRAM_ALLOWED_USER_IDS)
    except ValueError as error:
        sys.exit(str(error))
    if not allowed:
        log.warning("TELEGRAM_ALLOWED_USER_IDS is empty: nobody can ask questions. Message the "
                    "bot to learn your user ID, add it to .env, and restart.")
    try:
        retrieval.warm_up()
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"The search index isn't ready: {error}")

    log.info("starting; %d allowed user(s)", len(allowed))
    application = build_application(config.TELEGRAM_BOT_TOKEN, Chattfinch(allowed))
    # drop_pending_updates: messages sent while the bot was down are discarded, so a
    # restart doesn't answer (and pay for) a backlog of stale questions.
    application.run_polling(allowed_updates=["message", "callback_query"],
                            drop_pending_updates=True)


if __name__ == "__main__":
    main()
