"""Citation handling: connect the [n] markers in an answer to the sources shown.

The model cites sources by number ([2], [1][3], sometimes [1, 3]). This module
finds those markers, checks them against the sources the model was really shown,
gives each source a readable label, and renders the answer and its source list
as Markdown. It is display logic only: no model, no retrieval.
"""

import re

from chattfinch.rag import Answer, Source

# [2], and [1, 3] / [1,3] in case the model groups numbers in one bracket.
MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

ORIGINS = {
    "wikipedia": "Wikipedia (CC BY-SA 4.0)",  # the license requires attribution
    "ebird": "eBird",
    "ebird_live": "eBird recent sightings",
    "iucn": "IUCN Red List",
}
# Origins of sources that were looked up rather than retrieved by search: they have
# no meaningful similarity score.
LOOKED_UP = {"ebird_live", "iucn"}


def cited_numbers(text: str) -> list[int]:
    """Every number cited in `text`, in order of first appearance, without repeats."""
    numbers = [int(n) for group in MARKER.findall(text) for n in group.split(",")]
    return list(dict.fromkeys(numbers))


def cited_sources(answer: Answer) -> list[Source]:
    """The sources the answer cites, in order of first citation."""
    by_number = {source.number: source for source in answer.sources}
    return [by_number[n] for n in cited_numbers(answer.text) if n in by_number]


def uncited_sources(answer: Answer) -> list[Source]:
    """Sources the model was shown but didn't cite, in retrieval order."""
    cited = set(cited_numbers(answer.text))
    return [source for source in answer.sources if source.number not in cited]


def invalid_citations(answer: Answer) -> list[int]:
    """Numbers the answer cites that match no source it was shown. Anything here
    means the model invented a citation, and the answer shouldn't be trusted as is."""
    shown = {source.number for source in answer.sources}
    return [n for n in cited_numbers(answer.text) if n not in shown]


def source_label(source: Source) -> str:
    """e.g. "Saker Falcon (Falco cherrug) — Ecology · Wikipedia (CC BY-SA 4.0)"."""
    return (f"{source.common_name} ({source.scientific_name}) — {source.section} · "
            f"{ORIGINS.get(source.origin, source.origin)}")


def link_citations(answer: Answer) -> str:
    """The answer text as Markdown, with each valid [n] turned into a link to its
    source. A number with no matching source is left as plain text."""
    urls = {source.number: source.url for source in answer.sources}

    def link(match: re.Match) -> str:
        numbers = [int(n) for n in match.group(1).split(",")]
        return "".join(f"[\\[{n}\\]]({urls[n]})" if n in urls else f"\\[{n}\\]"
                       for n in numbers)

    return MARKER.sub(link, answer.text)


def sources_markdown(sources: list[Source]) -> str:
    """A Markdown list of sources: number, label, link."""
    return "\n".join(f"- **\\[{source.number}\\]** [{source_label(source)}]({source.url})"
                     for source in sources)
