import pytest

from chattfinch import citations
from chattfinch.rag import Answer, Source


def source(number, common="Saker Falcon", scientific="Falco cherrug", section="Ecology",
           origin="wikipedia"):
    return Source(number=number, species_code="sakfal1", common_name=common,
                  scientific_name=scientific, section=section, origin=origin,
                  url=f"https://example.org/{number}", score=0.7, text=f"Text of source {number}.")


def answer(text, numbers=(1, 2, 3)):
    return Answer(query="q", text=text, declined=False, sources=[source(n) for n in numbers])


@pytest.mark.parametrize("text, expected", [
    ("It hunts squirrels [1].", [1]),
    ("A [2]. B [1]. C [2].", [2, 1]),  # order of first appearance, no repeats
    ("Both agree [1][3].", [1, 3]),
    ("Grouped [1, 3] and [2,4].", [1, 3, 2, 4]),
    ("No citations here.", []),
    ("Not citations: [a], [], [1a], [ 1 ], 1], [see 2].", []),
    ("Multi-digit [12] works.", [12]),
])
def test_cited_numbers(text, expected):
    assert citations.cited_numbers(text) == expected


def test_cited_and_uncited_sources_partition_what_was_shown():
    result = answer("B [3]. A [1]. B again [3].")
    assert [s.number for s in citations.cited_sources(result)] == [3, 1]  # citation order
    assert [s.number for s in citations.uncited_sources(result)] == [2]  # retrieval order
    assert citations.invalid_citations(result) == []


def test_invented_citation_numbers_are_reported_not_resolved():
    result = answer("Real [1]. Invented [7] and [0].")
    assert citations.invalid_citations(result) == [7, 0]
    assert [s.number for s in citations.cited_sources(result)] == [1]
    assert [s.number for s in citations.uncited_sources(result)] == [2, 3]


def test_declined_answer_with_no_citations_cites_nothing():
    result = Answer(query="q", text="Nothing about penguins.", declined=True,
                    sources=[source(1), source(2)])
    assert citations.cited_sources(result) == []
    assert [s.number for s in citations.uncited_sources(result)] == [1, 2]


def test_source_label_names_species_section_and_origin():
    assert citations.source_label(source(1)) == (
        "Saker Falcon (Falco cherrug) — Ecology · Wikipedia (CC BY-SA 4.0)")
    assert citations.source_label(source(1, section="Profile", origin="ebird")).endswith(
        "— Profile · eBird")
    assert citations.source_label(source(1, origin="other")).endswith("· other")


def test_link_citations_turns_markers_into_links():
    linked = citations.link_citations(answer("Squirrels [1]. Pikas too [2][3]."))
    assert linked == ("Squirrels [\\[1\\]](https://example.org/1). Pikas too "
                      "[\\[2\\]](https://example.org/2)[\\[3\\]](https://example.org/3).")


def test_link_citations_splits_grouped_markers_and_leaves_invalid_ones_unlinked():
    linked = citations.link_citations(answer("Grouped [1, 9]."))
    assert linked == "Grouped [\\[1\\]](https://example.org/1)\\[9\\]."


def test_link_citations_leaves_other_brackets_alone():
    text = "The call is [a soft churr] per the source [1]."
    assert citations.link_citations(answer(text)).startswith("The call is [a soft churr] per")


def test_sources_markdown_lists_number_label_and_link():
    markdown = citations.sources_markdown([source(2), source(1, section="Status")])
    assert markdown.split("\n") == [
        "- **\\[2\\]** [Saker Falcon (Falco cherrug) — Ecology · Wikipedia (CC BY-SA 4.0)]"
        "(https://example.org/2)",
        "- **\\[1\\]** [Saker Falcon (Falco cherrug) — Status · Wikipedia (CC BY-SA 4.0)]"
        "(https://example.org/1)"]
    assert citations.sources_markdown([]) == ""
