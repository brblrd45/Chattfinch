"""Turn an Answer into Telegram messages (HTML parse mode). Pure: no Telegram calls.

Telegram accepts a small HTML subset (<b>, <i>, <a href>), and rejects a message
whose markup doesn't parse, so every piece of text that isn't our own markup is
escaped. A message holds at most 4,096 characters; lengths here are measured on
the HTML, which is never shorter than the visible text, so the limit is safe.
"""

import html
import re

from chattfinch import citations
from chattfinch.rag import Answer, Source

MAX_MESSAGE = 4096
_TAG = re.compile(r"<[^>]+>")


def escape(text: str) -> str:
    return html.escape(text, quote=False)


def link(url: str, label_html: str) -> str:
    return f'<a href="{html.escape(url, quote=True)}">{label_html}</a>'


def linked_answer(answer: Answer) -> str:
    """The answer text, escaped, with each valid [n] linked to its source. A number
    that matches no source stays as plain text."""
    urls = {source.number: source.url for source in answer.sources}

    def replace(match: re.Match) -> str:
        numbers = [int(n) for n in match.group(1).split(",")]
        return "".join(link(urls[n], f"[{n}]") if n in urls else f"[{n}]" for n in numbers)

    return citations.MARKER.sub(replace, escape(answer.text))


def source_line(source: Source) -> str:
    return f"[{source.number}] {link(source.url, escape(citations.source_label(source)))}"


def pack(blocks: list[str], limit: int = MAX_MESSAGE, separator: str = "\n\n") -> list[str]:
    """Join blocks into as few messages as possible, each within `limit`. Blocks are
    never split (splitting could cut an HTML tag in half), so each must fit."""
    messages: list[str] = []
    for block in blocks:
        if len(block) > limit:
            raise ValueError(f"A {len(block)}-character block can't fit in a {limit}-character message")
        if messages and len(messages[-1]) + len(separator) + len(block) <= limit:
            messages[-1] += separator + block
        else:
            messages.append(block)
    return messages


def truncate(text: str, limit: int) -> str:
    """Plain text cut to `limit` characters, ending in an ellipsis if cut."""
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def escape_to_fit(text: str, limit: int) -> str:
    """`text` escaped and cut so the *escaped* result is within `limit` (escaping
    lengthens text: "&" becomes "&amp;")."""
    size = limit
    while len(escape(truncate(text, size))) > limit:
        size -= 50
    return escape(truncate(text, size))


def names_line(answer: Answer) -> str:
    """"Eurasian Nightjar (also: Common goatsucker)" for each bird the question named
    by another name, so the user sees which bird the answer is about; "" if none."""
    return "; ".join(f"<i>{escape(common)} (also: {escape(asked)})</i>"
                     for asked, common in answer.renamed)


def answer_messages(answer: Answer, limit: int = MAX_MESSAGE) -> list[str]:
    """The reply to a question: which bird a renamed one is, the answer with linked
    citations (or the message explaining why there is none), a caution if it cites
    a source it wasn't shown, then the cited sources. Normally one message; the
    sources move to a second one if together they'd be too long."""
    if answer.needs_choice:  # just the question; the bot adds a button per bird
        return [escape(answer.text)]
    # A declined answer's text is already the full message to show (see
    # rag.parse_reply); it goes through the same path, so a "detail missing" reply
    # that cites the passage it checked gets its link too.
    body = linked_answer(answer)
    if len(body) > limit:  # far longer than the prompt asks for; drop the links to cut safely
        body = escape_to_fit(answer.text, limit)

    names = names_line(answer)
    blocks = [names, body] if names and not answer.declined else [body]
    invalid = citations.invalid_citations(answer)
    if invalid:
        numbers = ", ".join(f"[{n}]" for n in invalid)
        blocks.append(f"⚠️ <i>The answer cites {numbers}, which matches no source it was given. "
                      "Treat it with caution.</i>")
    cited = citations.cited_sources(answer)
    if cited:
        lines = ["<b>Sources</b>"]
        for source in cited:
            if len("\n".join([*lines, source_line(source)])) > limit:
                break
            lines.append(source_line(source))
        blocks.append("\n".join(lines))
    return pack(blocks, limit)


def passage_block(source: Source, limit: int = MAX_MESSAGE) -> str:
    score = ("" if source.origin in citations.LOOKED_UP
             else f"Similarity to the question: {source.score:.2f}\n")
    head = f"<b>{source_line(source)}</b>\n{score}"
    return head + escape_to_fit(source.text, limit - len(head))


def passage_messages(answer: Answer, limit: int = MAX_MESSAGE) -> list[str]:
    """Every passage the model was shown, cited ones first, packed into messages."""
    cited, uncited = citations.cited_sources(answer), citations.uncited_sources(answer)
    blocks = [passage_block(source, limit) for source in cited]
    if uncited:
        blocks.append("<i>Retrieved but not cited in the answer:</i>")
        blocks += [passage_block(source, limit) for source in uncited]
    return pack(blocks, limit)


def plain(message_html: str) -> str:
    """The message with markup removed, for resending if Telegram rejects the HTML."""
    return html.unescape(_TAG.sub("", message_html))
