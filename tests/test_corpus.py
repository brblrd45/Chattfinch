import gzip

import pandas as pd

from chattfinch import corpus

SPECIES = pd.DataFrame({
    "species_code": ["eurnig1", "egynig1"],
    "common_name": ["Eurasian Nightjar", "Egyptian Nightjar"],
    "scientific_name": ["Caprimulgus europaeus", "Caprimulgus aegyptius"],
    "order": ["Caprimulgiformes"] * 2,
    "family": ["Caprimulgidae"] * 2,
    "family_common_name": ["Nightjars and Allies"] * 2,
})

ALIASES = pd.DataFrame({
    "normalized": ["eurasian nightjar", "caprimulgus europaeus", "common goatsucker",
                   "european nightjar", "nightjar", "nightjar"],
    "alias": ["Eurasian Nightjar", "Caprimulgus europaeus", "Common goatsucker",
              "European nightjar", "Nightjar", "Nightjar"],
    "species_code": ["eurnig1", "eurnig1", "eurnig1", "eurnig1", "eurnig1", "egynig1"],
    "kind": ["species", "species", "species", "species", "group", "group"],
    "sources": ["ebird", "scientific", "wikipedia", "wikipedia|gbif", "group", "group"],
})

ARTICLE = {"title": "European nightjar", "revision_id": 42, "revision_timestamp": "2026-01-01T00:00:00Z",
           "url": "https://en.wikipedia.org/w/index.php?oldid=42",
           "sections": {"Introduction": "The European nightjar is...", "Behaviour > Feeding": "Moths."}}

RECORDS = {"eurnig1": {"article": ARTICLE, "article_fetched_on": "2026-10-08"},
           "egynig1": {"article": None}}


def build():
    return corpus.build_corpus(SPECIES, RECORDS, ALIASES)


def test_document_fields_and_attribution():
    doc = build()[0]
    assert doc["species_code"] == "eurnig1" and doc["family"] == "Caprimulgidae"
    assert doc["wikipedia"] == {"title": "European nightjar", "revision_id": 42,
                                "revision_timestamp": "2026-01-01T00:00:00Z",
                                "url": "https://en.wikipedia.org/w/index.php?oldid=42",
                                "retrieved_on": "2026-10-08", "license": "CC BY-SA 4.0"}


def test_sections_are_source_tagged():
    sections = build()[0]["sections"]
    assert [(s["source"], s["section"]) for s in sections] == [
        ("ebird", "Profile"), ("wikipedia", "Introduction"), ("wikipedia", "Behaviour > Feeding")]
    assert sections[0]["url"] == "https://ebird.org/species/eurnig1"
    assert sections[2]["url"] == ARTICLE["url"]


def test_profile_lists_species_aliases_but_not_group_names():
    doc = build()[0]
    assert doc["aliases"] == ["Common goatsucker", "European nightjar"]
    profile = doc["sections"][0]["text"]
    assert profile.startswith("Eurasian Nightjar (Caprimulgus europaeus) is a bird species in "
                              "the family Caprimulgidae (Nightjars and Allies)")
    assert "Kazakhstan" in profile
    assert profile.endswith("Also known as: Common goatsucker; European nightjar.")


def test_species_without_article_gets_profile_only_and_is_a_miss():
    docs = build()
    assert docs[1]["wikipedia"] is None
    assert [s["source"] for s in docs[1]["sections"]] == ["ebird"]
    assert "Also known as" not in docs[1]["sections"][0]["text"]
    assert corpus.coverage_misses(docs) == ["egynig1"]


def test_write_is_deterministic_and_round_trips(tmp_path):
    docs = build()
    first, second = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    corpus.write_corpus(docs, first)
    corpus.write_corpus(docs, second)
    assert first.read_bytes() == second.read_bytes()
    assert corpus.read_corpus(first) == docs
    with gzip.open(first, "rt") as f:
        assert len(f.readlines()) == 2
