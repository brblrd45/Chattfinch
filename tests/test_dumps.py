"""Checks on the committed dumps in data/dumps/ (the plan's day-1 data tests).
Skipped if the dumps haven't been built."""

import json

import pandas as pd
import pytest

from chattfinch import config, corpus
from chattfinch.aliases import normalize

DUMPS = config.DUMPS_DIR
pytestmark = pytest.mark.skipif(not (DUMPS / "alias_index.json").exists(),
                                reason="dumps not built; run scripts/build_dumps.py")


@pytest.fixture(scope="module")
def species():
    return pd.read_csv(DUMPS / "species_list.csv")


@pytest.fixture(scope="module")
def aliases():
    return pd.read_csv(DUMPS / "aliases.csv")


@pytest.fixture(scope="module")
def index():
    return json.loads((DUMPS / "alias_index.json").read_text())


def test_species_codes_unique(species):
    assert species["species_code"].is_unique


def test_every_species_reachable_by_ebird_and_scientific_name(species, index):
    for row in species.itertuples():
        assert index[normalize(row.common_name)] == [row.species_code]
        assert row.species_code in index[normalize(row.scientific_name)]


@pytest.mark.parametrize("names, code", [
    (["Eurasian Nightjar", "European Nightjar", "Goatsucker"], "eurnig1"),
    (["Sociable Lapwing", "Sociable Plover"], "soclap1"),
    (["Demoiselle Crane", "Anthropoides virgo", "Grus virgo"], "demcra1"),
    (["Black-bellied Plover", "Grey Plover"], "bkbplo"),
])
def test_known_name_pairs_resolve_to_same_species(index, names, code):
    for name in names:
        assert index[normalize(name)] == [code], name


def test_ambiguous_aliases_are_groups_or_reviewed(aliases, index):
    """Every alias mapping to 2+ species is either a group name ("warbler") or on the
    reviewed list below. A new ambiguous name fails this test until it's reviewed:
    add it here if the ambiguity is genuine, or fix it in alias_overrides.csv."""
    reviewed = {
        "northern grey shrike",  # pre-split name for Great Gray + Northern Shrike
        "red rumped swallow",  # pre-split name for European + Eastern Red-rumped Swallow
        "cat owl",  # folk name for both Long-eared Owl and Tawny Owl
        "pond bird",  # GBIF junk for Ruddy Turnstone + Sanderling; harmless
    }
    groups = set(aliases.loc[aliases["kind"] == "group", "normalized"])
    unreviewed = {a for a, codes in index.items() if len(codes) > 1} - groups - reviewed
    assert not unreviewed, sorted(unreviewed)


@pytest.fixture(scope="module")
def corpus_docs():
    path = DUMPS / "corpus.jsonl.gz"
    if not path.exists():
        pytest.skip("corpus not built")
    return corpus.read_corpus(path)


def test_corpus_has_one_document_per_species(species, corpus_docs):
    assert [d["species_code"] for d in corpus_docs] == list(species["species_code"])


def test_corpus_coverage(corpus_docs):
    """Every species has Wikipedia text with a revision permalink (the plan's day-1
    coverage check). A miss fails with the list of species codes."""
    assert corpus.coverage_misses(corpus_docs) == []
    for doc in corpus_docs:
        assert doc["wikipedia"]["url"].startswith("https://en.wikipedia.org/w/index.php?oldid=")
        assert doc["wikipedia"]["license"] == "CC BY-SA 4.0"


def test_dumps_fit_the_storage_budget():
    """The committed dumps (not the regenerated vector store) stay within
    config.DUMPS_BUDGET_MB. Raise the budget deliberately, or shrink the dumps."""
    size_mb = sum(f.stat().st_size for f in DUMPS.iterdir() if f.is_file()) / 1e6
    assert size_mb <= config.DUMPS_BUDGET_MB, f"{size_mb:.1f} MB > {config.DUMPS_BUDGET_MB} MB"
