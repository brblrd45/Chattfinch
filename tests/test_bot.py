"""The bot's behavior with fake Telegram updates and a fake answer pipeline:
no Telegram connection, no model, no index."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest

from chattfinch import bot as bot_module
from chattfinch import config, llm, prompts, rag, router
from chattfinch.bot import AnswerCache, Chattfinch, RateLimiter, parse_user_ids, question_key
from chattfinch.resolve import Match
from test_telegram_format import source

ALLOWED, STRANGER = 111, 999


class FakeMessage:
    """Stands in for telegram.Message: records what the bot replies."""

    def __init__(self, text=None, reject_html=False):
        self.text, self.reject_html = text, reject_html
        self.replies, self.actions = [], []

    async def reply_text(self, text, parse_mode=None, link_preview_options=None, reply_markup=None):
        if self.reject_html and parse_mode == ParseMode.HTML:
            raise BadRequest("Can't parse entities")
        self.replies.append(SimpleNamespace(text=text, parse_mode=parse_mode, markup=reply_markup,
                                            previews=link_preview_options))

    async def reply_chat_action(self, action):
        self.actions.append(action)

    @property
    def texts(self):
        return [reply.text for reply in self.replies]


class FakeQuery:
    def __init__(self, data):
        self.data, self.answered, self.markup_removed = data, 0, False

    async def answer(self):
        self.answered += 1

    async def edit_message_reply_markup(self, reply_markup=None):
        self.markup_removed = reply_markup is None


def update(user_id=ALLOWED, text=None, data=None, **message_options):
    return SimpleNamespace(effective_user=SimpleNamespace(id=user_id) if user_id else None,
                           effective_message=FakeMessage(text, **message_options),
                           callback_query=FakeQuery(data) if data is not None else None)


class Pipeline:
    """A fake rag.answer: records questions, returns `.result` (or raises it)."""

    def __init__(self):
        self.questions, self.threads, self.choices = [], [], []
        self.result = rag.Answer(query="q", text="It hunts ground squirrels [1].", declined=False,
                                 sources=[source(1), source(2)])

    def __call__(self, question, choice=None):
        self.questions.append(question)
        self.choices.append(choice)
        self.threads.append(threading.current_thread())
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def pipeline():
    return Pipeline()


@pytest.fixture
def chirp(pipeline):
    return Chattfinch(frozenset({ALLOWED}), answer_fn=pipeline)


def run(coroutine):
    return asyncio.run(coroutine)


def buttons(reply):
    return [(b.text, b.callback_data) for row in reply.markup.inline_keyboard for b in row]


# --- answering ---------------------------------------------------------------------

def test_question_gets_an_html_answer_with_a_show_passages_button(chirp, pipeline):
    u = update(text="  What does the Saker Falcon eat?  ")
    run(chirp.question(u))
    assert pipeline.questions == ["What does the Saker Falcon eat?"]  # stripped
    (reply,) = u.effective_message.replies
    assert reply.parse_mode == ParseMode.HTML
    assert 'ground squirrels <a href="https://example.org/1">[1]</a>' in reply.text
    assert "<b>Sources</b>" in reply.text
    assert reply.previews.is_disabled  # no link preview cards under the answer
    assert buttons(reply) == [("Show passages", f"p:{question_key('What does the Saker Falcon eat?')}")]
    assert u.effective_message.actions == [ChatAction.TYPING]


def test_the_pipeline_runs_off_the_event_loop_thread(chirp, pipeline):
    run(chirp.question(update(text="A question")))
    assert pipeline.threads[0] is not threading.main_thread()


def test_bird_not_found_gets_the_fixed_message_and_no_passages_button(chirp, pipeline):
    pipeline.result = rag.Answer(query="q", text=rag.DECLINE_MESSAGES[prompts.BIRD_NOT_FOUND],
                                 declined=True, sources=[source(1)],
                                 decline_reason=prompts.BIRD_NOT_FOUND)
    u = update(text="Where does the tufted titmouse live?")
    run(chirp.question(u))
    (reply,) = u.effective_message.replies
    assert reply.text == ("I couldn't find that bird in my sources. I only cover birds recorded "
                          "in Kazakhstan, so it may not be one of them.")
    assert reply.markup is None  # the retrieved passages are about other birds


def test_detail_missing_keeps_the_passages_button(chirp, pipeline):
    pipeline.result = rag.Answer(query="q", text="The sources don't give its lifespan.",
                                 declined=True, sources=[source(1)],
                                 decline_reason=prompts.DETAIL_MISSING)
    u = update(text="How long does the Saker Falcon live?")
    run(chirp.question(u))
    (reply,) = u.effective_message.replies
    assert reply.text == "The sources don't give its lifespan."
    assert buttons(reply)[0][0] == "Show passages"  # these are about the right bird


def test_answer_with_no_sources_has_no_button(chirp, pipeline):
    pipeline.result = rag.Answer(query="q", text=rag.NOTHING_RETRIEVED, declined=True, sources=[])
    u = update(text="Anything?")
    run(chirp.question(u))
    assert u.effective_message.replies[0].markup is None


def test_long_answer_sends_two_messages_with_the_button_on_the_last(chirp, pipeline):
    pipeline.result = rag.Answer(query="q", text="Fact [1]. " + "Padding sentence. " * 220,
                                 declined=False, sources=[source(1)])
    u = update(text="A long one")
    run(chirp.question(u))
    first, second = u.effective_message.replies
    assert first.markup is None and second.text.startswith("<b>Sources</b>")
    assert buttons(second)[0][0] == "Show passages"


@pytest.mark.parametrize("failure", [llm.LLMError("No valid Anthropic credentials"),
                                     RuntimeError("index exploded")])
def test_failures_get_an_apology_without_details(chirp, pipeline, failure):
    pipeline.result = failure
    u = update(text="A question")
    run(chirp.question(u))
    assert u.effective_message.texts == [bot_module.FAILED]
    assert "credentials" not in u.effective_message.texts[0]  # internals stay in the log


def test_failed_answers_are_not_cached(chirp, pipeline):
    pipeline.result = llm.LLMError("temporary")
    run(chirp.question(update(text="Same question")))
    pipeline.result = rag.Answer(query="q", text="Now it works [1].", declined=False,
                                 sources=[source(1)])
    u = update(text="Same question")
    run(chirp.question(u))
    assert len(pipeline.questions) == 2 and "Now it works" in u.effective_message.texts[0]


@pytest.mark.parametrize("text, expected", [("", bot_module.EMPTY), ("   \n ", bot_module.EMPTY),
                                            (None, bot_module.EMPTY)])
def test_empty_message_asks_for_a_question_without_calling_the_model(chirp, pipeline, text, expected):
    u = update(text=text)
    run(chirp.question(u))
    assert u.effective_message.texts == [expected] and pipeline.questions == []


def test_overlong_question_is_refused_without_calling_the_model(chirp, pipeline):
    u = update(text="x" * (config.TELEGRAM_MAX_QUESTION_CHARS + 1))
    run(chirp.question(u))
    assert str(config.TELEGRAM_MAX_QUESTION_CHARS) in u.effective_message.texts[0]
    assert pipeline.questions == []


def test_html_rejected_by_telegram_is_resent_as_plain_text(chirp):
    u = update(text="A question", reject_html=True)
    run(chirp.question(u))
    (reply,) = u.effective_message.replies
    assert reply.parse_mode is None
    assert reply.text.startswith("It hunts ground squirrels [1].") and "<a href" not in reply.text
    assert buttons(reply)[0][0] == "Show passages"  # the button survives the fallback


# --- cache and rate limit ----------------------------------------------------------

def test_repeated_question_is_served_from_the_cache(chirp, pipeline):
    run(chirp.question(update(text="What does it eat?")))
    again = update(text="  what DOES it   eat? ")  # same question, different case and spacing
    run(chirp.question(again))
    assert len(pipeline.questions) == 1
    assert "ground squirrels" in again.effective_message.texts[0]
    assert again.effective_message.actions == []  # no "typing…" for an instant reply


def test_rate_limit_stops_new_questions_but_not_cached_ones(pipeline):
    chirp = Chattfinch(frozenset({ALLOWED}), answer_fn=pipeline, rate_limiter=RateLimiter(2, 3600))
    run(chirp.question(update(text="one")))
    run(chirp.question(update(text="two")))
    third = update(text="three")
    run(chirp.question(third))
    assert third.effective_message.texts == [bot_module.RATE_LIMITED]
    assert pipeline.questions == ["one", "two"]
    cached = update(text="one")
    run(chirp.question(cached))
    assert "ground squirrels" in cached.effective_message.texts[0]


def test_rate_limiter_is_per_user_and_recovers_after_the_period():
    now = [0.0]
    limiter = RateLimiter(2, 60, clock=lambda: now[0])
    assert limiter.allow(1) and limiter.allow(1) and not limiter.allow(1)
    assert limiter.allow(2)  # another user is unaffected
    now[0] = 59.9
    assert not limiter.allow(1)
    now[0] = 60.0
    assert limiter.allow(1)  # the first two calls have aged out
    assert limiter.allow(1) and not limiter.allow(1)


def test_refused_calls_do_not_extend_the_wait():
    now = [0.0]
    limiter = RateLimiter(1, 60, clock=lambda: now[0])
    assert limiter.allow(1)
    now[0] = 30.0
    assert not limiter.allow(1)  # refused: must not count as a call
    now[0] = 60.0
    assert limiter.allow(1)


def test_answer_cache_evicts_the_least_recently_used():
    cache = AnswerCache(max_entries=2)
    first, second, third = (rag.Answer(query=str(n), text="", declined=True, sources=[])
                            for n in range(3))
    cache.put("a", first)
    cache.put("b", second)
    assert cache.get("a") is first  # touching "a" makes "b" the oldest
    cache.put("c", third)
    assert cache.get("b") is None and cache.get("a") is first and cache.get("c") is third


def test_question_key_ignores_case_punctuation_and_spacing_and_fits_in_button_data():
    assert question_key("What does it eat?") == question_key("  what DOES it   eat ")
    assert question_key("Pallas's Sandgrouse, where?") == question_key("pallas s sandgrouse where")
    assert question_key("What does it eat?") != question_key("What does it hunt?")
    assert len(f"p:{question_key('x' * 500)}".encode()) <= 64  # Telegram's button data limit


def test_question_key_keeps_letters_in_any_script():
    """aliases.normalize drops non-Latin letters; the cache key must not, or every
    Cyrillic question would share one key."""
    assert question_key("Что ест сапсан?") != question_key("Где живёт дрофа?")
    assert question_key("Что ест сапсан?") == question_key("что ест САПСАН")


# --- access control ----------------------------------------------------------------

@pytest.mark.parametrize("handler, kwargs", [
    ("question", {"text": "What does the Saker Falcon eat?"}),
    ("start", {"text": "/start"}),
    ("unsupported", {}),
    ("button", {"data": "ex:0"}),
    ("button", {"data": "p:abc"}),
])
def test_stranger_is_refused_everywhere_and_never_reaches_the_model(chirp, pipeline, handler, kwargs):
    u = update(user_id=STRANGER, **kwargs)
    run(getattr(chirp, handler)(u))
    assert u.effective_message.texts == [f"This bot is private. Your Telegram user ID is {STRANGER}."]
    assert pipeline.questions == [] and u.effective_message.actions == []
    if u.callback_query:
        assert u.callback_query.answered == 1  # the button's spinner still stops


def test_update_without_a_user_is_refused(chirp, pipeline):
    u = update(user_id=None, text="A question")
    run(chirp.question(u))
    assert "private" in u.effective_message.texts[0] and pipeline.questions == []


def test_empty_allowlist_refuses_everyone(pipeline):
    chirp = Chattfinch(frozenset(), answer_fn=pipeline)
    u = update(text="A question")
    run(chirp.question(u))
    assert "private" in u.effective_message.texts[0] and pipeline.questions == []


@pytest.mark.parametrize("value, expected", [
    ("111", {111}), (" 111 , 222,333 ", {111, 222, 333}), ("", set()), (" , ", set()),
    ("111,111", {111}),
])
def test_parse_user_ids(value, expected):
    assert parse_user_ids(value) == frozenset(expected)


@pytest.mark.parametrize("value", ["@username", "111;222", "12a", "-5", "1.5"])
def test_parse_user_ids_rejects_anything_but_numbers(value):
    with pytest.raises(ValueError, match="TELEGRAM_ALLOWED_USER_IDS"):
        parse_user_ids(value)


# --- commands and buttons ----------------------------------------------------------

def test_start_explains_the_bot_and_offers_examples(chirp, pipeline):
    u = update(text="/start")
    run(chirp.start(u))
    (reply,) = u.effective_message.replies
    assert "Ask a Birder" in reply.text and config.REGION_NAME in reply.text
    assert buttons(reply) == [(example, f"ex:{i}") for i, example in enumerate(bot_module.EXAMPLES)]
    assert pipeline.questions == []


def test_example_button_asks_that_question(chirp, pipeline):
    u = update(data="ex:1")
    run(chirp.button(u))
    assert pipeline.questions == [bot_module.EXAMPLES[1]]
    assert u.callback_query.answered == 1
    assert "ground squirrels" in u.effective_message.texts[0]


@pytest.mark.parametrize("data", ["ex:99", "ex:abc", "ex:", "zz:1", "", "ex:-1"])
def test_malformed_button_data_is_ignored(chirp, pipeline, data):
    u = update(data=data)
    run(chirp.button(u))
    assert pipeline.questions == [] and u.effective_message.replies == []
    assert u.callback_query.answered == 1


def test_show_passages_sends_the_passages_and_removes_the_button(chirp, pipeline):
    asked = update(text="What does it eat?")
    run(chirp.question(asked))
    (_, data), = buttons(asked.effective_message.replies[0])
    pressed = update(data=data)
    run(chirp.button(pressed))
    (passages,) = pressed.effective_message.texts
    assert "Text of source 1." in passages and "Text of source 2." in passages
    assert "Retrieved but not cited" in passages  # source 2
    assert pressed.callback_query.markup_removed
    assert len(pipeline.questions) == 1  # showing passages costs nothing


def test_show_passages_for_an_expired_answer_says_so(chirp, pipeline):
    u = update(data="p:0123456789abcdef")
    run(chirp.button(u))
    assert u.effective_message.texts == [bot_module.EXPIRED]
    assert not u.callback_query.markup_removed


def test_non_text_message_gets_a_text_only_reply(chirp, pipeline):
    u = update()
    run(chirp.unsupported(u))
    assert u.effective_message.texts == [bot_module.TEXT_ONLY] and pipeline.questions == []


# --- wiring ------------------------------------------------------------------------

def test_application_builds_without_contacting_telegram(chirp):
    application = bot_module.build_application("123456:TEST-token", chirp)
    handlers = application.handlers[0]
    assert [type(h).__name__ for h in handlers] == [
        "CommandHandler", "MessageHandler", "MessageHandler", "MessageHandler",
        "CallbackQueryHandler"]
    assert application.update_processor.max_concurrent_updates > 1  # slow answers don't block


@pytest.mark.parametrize("token, allowed, message", [
    (None, "111", "TELEGRAM_BOT_TOKEN is not set"),
    ("123:abc", "not-a-number", "TELEGRAM_ALLOWED_USER_IDS"),
])
def test_main_exits_with_a_clear_message_on_bad_settings(monkeypatch, token, allowed, message):
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setattr(config, "TELEGRAM_ALLOWED_USER_IDS", allowed)
    with pytest.raises(SystemExit, match=message):
        bot_module.main()


def test_main_exits_clearly_when_the_index_is_missing(monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setattr(config, "TELEGRAM_ALLOWED_USER_IDS", "111")

    def no_index():
        raise FileNotFoundError("No index at data/index; run scripts/build_index.py")
    monkeypatch.setattr(bot_module.retrieval, "warm_up", no_index)
    with pytest.raises(SystemExit, match="build_index.py"):
        bot_module.main()


def test_main_keeps_the_bot_token_out_of_the_logs(monkeypatch):
    """The HTTP library logs request URLs at INFO, and Telegram URLs contain the token."""
    import logging
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", None)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.NOTSET)
    with pytest.raises(SystemExit):
        bot_module.main()
    for name in ("httpx", "httpcore"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING


# --- choosing between birds ------------------------------------------------------------

CAT_OWL = Match("Cat Owl", "cat owl", ("loeowl", "tawowl1"), False, ("Long-eared Owl", "Tawny Owl"))
CHOOSING = rag.Answer(query="Is the cat owl nocturnal?", text="Which bird do you mean by “Cat Owl”?",
                      declined=False, sources=[], choices=CAT_OWL)


def test_an_ambiguous_name_gets_one_button_per_bird(chirp, pipeline):
    pipeline.result = CHOOSING
    u = update(text="Is the cat owl nocturnal?")
    run(chirp.question(u))
    (reply,) = u.effective_message.replies
    assert reply.text == "Which bird do you mean by “Cat Owl”?"
    key = question_key("Is the cat owl nocturnal?")
    assert buttons(reply) == [("Long-eared Owl", f"c:{key}:0"), ("Tawny Owl", f"c:{key}:1")]


def test_choosing_a_bird_answers_for_it_and_removes_the_buttons(chirp, pipeline):
    pipeline.result = CHOOSING
    run(chirp.question(update(text="Is the cat owl nocturnal?")))
    pipeline.result = rag.Answer(query="q", text="Yes [1].", declined=False, sources=[source(1)])
    key = question_key("Is the cat owl nocturnal?")
    u = update(data=f"c:{key}:1")
    run(chirp.button(u))
    assert pipeline.questions[-1] == "Is the cat owl nocturnal?"
    assert pipeline.choices == [None, "tawowl1"]
    assert u.callback_query.markup_removed
    (reply,) = u.effective_message.replies
    assert "Yes" in reply.text
    # The chosen answer is cached under its own key, so its passages can be shown.
    assert buttons(reply) == [("Show passages",
                               f"p:{question_key('Is the cat owl nocturnal?', 'tawowl1')}")]


@pytest.mark.parametrize("data", ["c:0123456789abcdef:0", "c:{key}:2", "c:{key}:x", "c:{key}"])
def test_an_expired_or_malformed_choice_asks_again(chirp, pipeline, data):
    pipeline.result = CHOOSING
    run(chirp.question(update(text="Is the cat owl nocturnal?")))
    u = update(data=data.format(key=question_key("Is the cat owl nocturnal?")))
    run(chirp.button(u))
    assert u.effective_message.texts == [bot_module.CHOICE_EXPIRED]
    assert pipeline.choices == [None]  # the model was never asked


def test_question_key_depends_on_the_choice():
    assert question_key("q", "tawowl1") != question_key("q") != question_key("q", "loeowl")
    assert question_key("Q ", "tawowl1") == question_key("q", "tawowl1")


# --- logging ---------------------------------------------------------------------------

def test_answer_log_has_the_decline_reason_and_name_counts_but_no_text(chirp, pipeline, caplog):
    goatsucker = Match("Common goatsucker", "common goatsucker", ("eurnig1",), False,
                       ("Eurasian Nightjar",))
    pipeline.result = rag.Answer(query="q", text="The sources don't say what it eats.",
                                 declined=True, sources=[source(1)],
                                 decline_reason=prompts.DETAIL_MISSING, names=(goatsucker,))
    with caplog.at_level("INFO", logger="chattfinch.bot"):
        run(chirp.question(update(text="What does the common goatsucker eat?")))
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("answered")]
    assert (f"user={ALLOWED} route=descriptive declined=True reason=detail_missing names=1 "
            "renamed=1 chosen=False") in line
    assert "goatsucker" not in caplog.text.lower() and "eats" not in caplog.text


def test_answer_log_without_a_decline_or_names(chirp, pipeline, caplog):
    with caplog.at_level("INFO", logger="chattfinch.bot"):
        run(chirp.question(update(text="A question")))
    assert "declined=False reason=- names=0 renamed=0 chosen=False" in caplog.text


def test_choices_and_passages_are_logged(chirp, pipeline, caplog):
    pipeline.result = CHOOSING
    run(chirp.question(update(text="Is the cat owl nocturnal?")))
    pipeline.result = rag.Answer(query="q", text="Yes [1].", declined=False, sources=[source(1)],
                                 names=(CAT_OWL.choose("tawowl1"),))
    with caplog.at_level("INFO", logger="chattfinch.bot"):
        run(chirp.button(update(data=f"c:{question_key('Is the cat owl nocturnal?')}:1")))
        key = question_key("Is the cat owl nocturnal?", "tawowl1")
        run(chirp.button(update(data=f"p:{key}")))
    assert "names=1 renamed=1 chosen=True" in caplog.text
    assert f"passages user={ALLOWED} sources=1" in caplog.text


# --- route-aware caching ---------------------------------------------------------------

class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def routed(route=router.DESCRIPTIVE, reason=None, declined=False):
    return rag.Answer(query="q", text="text", declined=declined, sources=[source(1)],
                      decline_reason=reason, route=route)


def test_live_answers_expire_but_others_stay():
    clock = Clock()
    cache = AnswerCache(live_ttl_s=600, clock=clock)
    live, status, descriptive = routed(router.LIVE), routed(router.STATUS), routed()
    cache.put("live", live)
    cache.put("status", status)
    cache.put("descriptive", descriptive)
    clock.now += 599
    assert cache.get("live") is live  # still there for "Show passages"
    clock.now += 1
    assert cache.get("live") is None
    clock.now += 10 ** 6
    assert cache.get("status") is status and cache.get("descriptive") is descriptive


@pytest.mark.parametrize("reason", [router.LIVE_UNAVAILABLE, router.STATUS_UNAVAILABLE])
def test_temporary_failures_are_not_cached(reason):
    cache = AnswerCache()
    cache.put("k", routed(router.LIVE, reason=reason, declined=True))
    assert cache.get("k") is None


def test_a_failure_replaces_an_older_cached_answer():
    cache = AnswerCache()
    cache.put("k", routed(router.LIVE))
    cache.put("k", routed(router.LIVE, reason=router.LIVE_UNAVAILABLE, declined=True))
    assert cache.get("k") is None


def test_a_repeated_live_question_fetches_again_after_the_ttl(pipeline):
    clock = Clock()
    chirp = Chattfinch(frozenset({ALLOWED}), answer_fn=pipeline,
                     cache=AnswerCache(live_ttl_s=600, clock=clock))
    pipeline.result = routed(router.LIVE)
    for minutes in (0, 5, 11):
        clock.now = 1000.0 + 60 * minutes
        run(chirp.question(update(text="Has the Sociable Lapwing been seen recently?")))
    assert len(pipeline.questions) == 2  # the 5-minute repeat came from the cache


def test_an_ebird_failure_is_retried_on_the_next_ask(chirp, pipeline):
    pipeline.result = routed(router.LIVE, reason=router.LIVE_UNAVAILABLE, declined=True)
    run(chirp.question(update(text="Seen recently?")))
    run(chirp.question(update(text="Seen recently?")))
    assert len(pipeline.questions) == 2
