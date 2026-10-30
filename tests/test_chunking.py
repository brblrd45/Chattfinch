import pytest

from chattfinch import chunking
from chattfinch.chunking import SPECIAL_TOKENS, chunk_document, pack

count = lambda text: len(text.split())  # one token per word: no model needed


def sentences(n: int, words_each: int = 10, start: int = 0) -> list[str]:
    """n sentences of `words_each` unique words, e.g. 'W0_1 w0_2 ... w0_10.' (the first
    word is capitalized because the sentence splitter only breaks before a capital)."""
    return [" ".join(f"{'W' if j == 1 else 'w'}{start + i}_{j}" for j in range(1, words_each + 1)) + "."
            for i in range(n)]


def doc(*sections: tuple[str, str]) -> dict:
    return {"species_code": "aaa1", "common_name": "A Bird", "scientific_name": "Genus a",
            "sections": [{"source": "wikipedia", "section": name, "text": text,
                          "url": "https://example.org/a"} for name, text in sections]}


def chunks_of(document, max_tokens=60, overlap=0):
    return chunk_document(document, count, max_tokens, overlap)


def test_every_chunk_fits_and_n_tokens_includes_special_tokens():
    body = " ".join(sentences(40))
    chunks = chunks_of(doc(("Diet", body)), max_tokens=60)
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.n_tokens == count(chunk.text) + SPECIAL_TOKENS
        assert chunk.n_tokens <= 60


def test_header_names_species_and_section_in_every_chunk():
    chunks = chunks_of(doc(("Behaviour > Feeding", " ".join(sentences(20)))))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("A Bird (Genus a) — Behaviour > Feeding\n")


def test_chunk_metadata_and_ids():
    chunks = chunks_of(doc(("One", " ".join(sentences(12))), ("Two", " ".join(sentences(12, start=50)))))
    assert {c.species_code for c in chunks} == {"aaa1"}
    assert {c.url for c in chunks} == {"https://example.org/a"}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for section in ("One", "Two"):
        assert [c.chunk_index for c in chunks if c.section == section] == list(
            range(sum(c.section == section for c in chunks)))


def test_sentences_are_never_split_when_they_fit():
    original = sentences(30)
    chunks = chunks_of(doc(("S", " ".join(original))))
    for sentence in original:
        assert any(sentence in chunk.text for chunk in chunks), sentence


def test_nothing_is_lost_without_overlap():
    original = sentences(30)
    bodies = [c.text.split("\n", 1)[1] for c in chunks_of(doc(("S", " ".join(original))))]
    assert " ".join(bodies) == " ".join(original)


def test_short_boundary_sentence_repeats_in_next_chunk():
    bodies = pack(" ".join(sentences(10)), budget=35, overlap=10, count=count)
    assert len(bodies) > 1
    for before, after in zip(bodies, bodies[1:]):
        last_sentence = before.split(". ")[-1].rstrip(".") + "."
        assert after.startswith(last_sentence)


def test_long_boundary_sentence_is_not_repeated():
    bodies = pack(" ".join(sentences(10)), budget=35, overlap=5, count=count)  # sentences are 10
    for before, after in zip(bodies, bodies[1:]):
        assert not set(before.split()) & set(after.split())


def test_overlap_never_pushes_a_chunk_over_budget():
    for budget in (25, 30, 35, 40):
        for body in pack(" ".join(sentences(30)), budget=budget, overlap=12, count=count):
            assert count(body) <= budget


def test_oversized_sentence_split_at_semicolons_then_words():
    aliases = "; ".join(f"Alias Number {i}" for i in range(40)) + "."  # one 120-word sentence
    bodies = pack(f"Also known as: {aliases}", budget=30, overlap=0, count=count)
    assert len(bodies) > 3 and all(count(b) <= 30 for b in bodies)
    assert all(f"Alias Number {i}" in " ".join(bodies) for i in range(40))
    assert sum(b.count(";") for b in bodies) >= 30  # mostly split at semicolons, not mid-alias
    words = " ".join(" ".join(bodies).split())
    assert words.split() == f"Also known as: {aliases}".split()  # no overlap: nothing lost


def test_sentence_with_no_break_points_is_split_at_words():
    text = " ".join(f"word{i}" for i in range(100))  # no punctuation at all
    bodies = pack(text, budget=30, overlap=0, count=count)
    assert [count(b) for b in bodies] == [30, 30, 30, 10]


def test_paragraphs_are_separated_not_merged_mid_sentence():
    bodies = pack("First line without a stop\nSecond line here.", budget=100, overlap=0, count=count)
    assert bodies == ["First line without a stop Second line here."]


def test_empty_sections_produce_no_chunks():
    assert chunks_of(doc(("Empty", "  \n "))) == []


def test_small_section_is_one_chunk():
    chunks = chunks_of(doc(("Status", "Least concern.")))
    assert [c.text for c in chunks] == ["A Bird (Genus a) — Status\nLeast concern."]


def test_header_that_leaves_no_room_raises():
    document = doc((" ".join(f"long{i}" for i in range(50)), "Some text."))
    with pytest.raises(ValueError, match="no room"):
        chunks_of(document, max_tokens=60)


def test_chunk_corpus_concatenates_documents():
    docs = [doc(("S", " ".join(sentences(5)))), {**doc(("S", " ".join(sentences(5)))),
                                                  "species_code": "bbb1"}]
    chunks = chunking.chunk_corpus(docs, count, 60, 0)
    assert [c.species_code for c in chunks] == ["aaa1", "bbb1"]


def test_sentence_splitter_does_not_break_before_lowercase():
    # "c. 1758" and "ca. 25" must not be read as sentence ends.
    body = "It was described by Linnaeus in c. 1758 and measures ca. 25 cm. Males are larger."
    assert pack(body, budget=100, overlap=0, count=count) == [body]
    assert [text for text, _ in chunking._units(body, 100, count)] == [
        "It was described by Linnaeus in c. 1758 and measures ca. 25 cm.", "Males are larger."]
