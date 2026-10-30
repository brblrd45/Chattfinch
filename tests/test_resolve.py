"""Name resolution: rules on a small synthetic alias table, then the committed table."""

import pandas as pd
import pytest

from chattfinch import aliases, resolve
from chattfinch.resolve import AMBIGUOUS, GROUP, SPECIES, Match, NameIndex

SPECIES_LIST = pd.DataFrame([
    ("eurnig1", "Eurasian Nightjar", "Caprimulgus europaeus"),
    ("eueowl1", "Eurasian Eagle-Owl", "Bubo bubo"),
    ("steeag1", "Steppe Eagle", "Aquila nipalensis"),
    ("loeowl", "Long-eared Owl", "Asio otus"),
    ("tawowl1", "Tawny Owl", "Strix aluco"),
    ("perfal", "Peregrine Falcon", "Falco peregrinus"),
    ("comswi", "Common Swift", "Apus apus"),
    ("alpswi1", "Alpine Swift", "Tachymarptis melba"),
    ("merlin", "Merlin", "Falco columbarius"),
    ("litter1", "Little Tern", "Sternula albifrons"),
    *[(f"war{i}", f"Warbler {i}", f"Sylvia w{i}") for i in range(6)],
], columns=["species_code", "common_name", "scientific_name"])

ROWS = [  # (alias, species, kind, sources)
    ("Eurasian Nightjar", "eurnig1", "species", "ebird|gbif|wikipedia"),
    ("Common goatsucker", "eurnig1", "species", "wikipedia"),
    ("Caprimulgus europaeus", "eurnig1", "species", "scientific|wikipedia"),
    ("Eurasian Eagle-Owl", "eueowl1", "species", "ebird"),
    ("Steppe Eagle", "steeag1", "species", "ebird"),
    ("Eagle", "eueowl1", "group", "group"), ("Eagle", "steeag1", "group", "group"),
    ("Owl", "eueowl1", "group", "group"), ("Owl", "loeowl", "group", "group"),
    ("Cat Owl", "loeowl", "species", "wikipedia"), ("Cat Owl", "tawowl1", "species", "gbif"),
    ("Peregrine Falcon", "perfal", "species", "ebird"),
    ("Swift", "comswi", "group", "group"), ("Swift", "alpswi1", "group", "group"),
    ("Merlin", "merlin", "species", "ebird"),
    ("Little", "litter1", "species", "wikipedia"),
    *[("Leaf bird", f"war{i}", "species", "gbif") for i in range(6)],  # 6 species, not a group
]
EVERYDAY = pd.DataFrame([("little", "never", ""), ("swift", "alone", ""),
                         ("merlin", "alone", "")], columns=["word", "rule", "note"])


@pytest.fixture(scope="module")
def names():
    table = pd.DataFrame([(aliases.normalize(a), a, code, kind, sources)
                          for a, code, kind, sources in ROWS], columns=aliases.ALIAS_COLUMNS)
    return NameIndex(table, EVERYDAY, SPECIES_LIST)


def kinds(resolution):
    return [(m.normalized, m.kind) for m in resolution.matches]


def test_a_species_alias_resolves_to_its_species(names):
    r = names.resolve("What does the common goatsucker eat?")
    assert r.species == ("eurnig1",) and r.ambiguous is None and r.groups == ()
    (match,) = r.matches
    assert (match.alias, match.kind, match.common_names) == (
        "Common goatsucker", SPECIES, ("Eurasian Nightjar",))


def test_normalization_matches_case_plurals_and_punctuation(names):
    assert names.resolve("where do EURASIAN EAGLE OWLS nest").species == ("eueowl1",)
    assert names.resolve("caprimulgus europaeus?").species == ("eurnig1",)


def test_the_longest_alias_wins(names):
    r = names.resolve("Is the Eurasian Eagle-Owl an eagle or an owl?")
    assert kinds(r) == [("eurasian eagle owl", SPECIES), ("eagle", GROUP), ("owl", GROUP)]
    assert r.species == ("eueowl1",)  # groups never filter


def test_several_species_are_kept_in_order_without_repeats(names):
    r = names.resolve("Steppe Eagle vs Peregrine Falcon: does the steppe eagle win?")
    assert r.species == ("steeag1", "perfal")


def test_an_ambiguous_alias_is_flagged_and_does_not_filter(names):
    r = names.resolve("Is the cat owl nocturnal?")
    assert r.ambiguous.alias == "Cat Owl"
    assert r.ambiguous.species == ("loeowl", "tawowl1")
    assert r.ambiguous.common_names == ("Long-eared Owl", "Tawny Owl")
    assert r.species == ()


def test_more_than_four_candidates_is_a_group_not_a_question(names):
    (match,) = names.resolve("what is a leaf bird").matches
    assert len(match.species) == 6 and match.kind == GROUP


def test_unknown_names_resolve_to_nothing(names):
    assert names.resolve("Tell me about penguins").matches == ()
    assert names.resolve("").matches == ()


def test_never_words_never_match(names):
    assert names.resolve("What little birds live here?").matches == ()


def test_alone_words_yield_to_any_other_bird(names):
    r = names.resolve("How swift is the Peregrine Falcon in a dive?")
    assert r.species == ("perfal",) and kinds(r) == [("peregrine falcon", SPECIES)]
    assert names.resolve("Merlin and the steppe eagle").species == ("steeag1",)


def test_alone_words_count_when_they_are_the_only_bird(names):
    assert names.resolve("Where does the merlin breed?").species == ("merlin",)
    assert kinds(names.resolve("How swift are swifts?")) == [("swift", GROUP), ("swift", GROUP)]


def test_choose_narrows_an_ambiguous_match():
    match = Match("Cat Owl", "cat owl", ("loeowl", "tawowl1"), False,
                  ("Long-eared Owl", "Tawny Owl"))
    chosen = match.choose("tawowl1")
    assert (chosen.species, chosen.kind, chosen.common_names) == (
        ("tawowl1",), SPECIES, ("Tawny Owl",))
    assert chosen.renamed == ("Cat Owl", "Tawny Owl")
    with pytest.raises(ValueError, match="not a candidate"):
        match.choose("perfal")


def test_renamed_only_when_the_name_differs_from_the_canonical_one(names):
    (goatsucker,) = names.resolve("common goatsucker").matches
    assert goatsucker.renamed == ("Common goatsucker", "Eurasian Nightjar")
    (nightjar,) = names.resolve("eurasian nightjars").matches
    assert nightjar.renamed is None
    (owl,) = names.resolve("owl").matches
    assert owl.renamed is None  # groups are never "renamed"


def test_display_form_prefers_tidier_sources():
    table = pd.DataFrame([("black necked grebe", "The Black-necked grebe", "eargre", "species",
                           "wikipedia"),
                          ("black necked grebe", "Black-necked Grebe", "eargre", "species",
                           "gbif|iucn")], columns=aliases.ALIAS_COLUMNS)
    species = pd.DataFrame([("eargre", "Eared Grebe", "Podiceps nigricollis")],
                           columns=SPECIES_LIST.columns)
    index = NameIndex(table, EVERYDAY.iloc[:0], species)
    assert index.entries["black necked grebe"].alias == "Black-necked Grebe"


def test_unknown_everyday_rule_is_rejected(names):
    table = pd.DataFrame([("merlin", "Merlin", "merlin", "species", "ebird")],
                         columns=aliases.ALIAS_COLUMNS)
    bad = pd.DataFrame([("merlin", "sometimes", "")], columns=["word", "rule", "note"])
    with pytest.raises(ValueError, match="unknown rules"):
        NameIndex(table, bad, SPECIES_LIST)


# --- The committed alias table and everyday-word list ----------------------------------

@pytest.mark.parametrize("names_of_one_species, code", [
    (["Eurasian Nightjar", "European Nightjar", "common goatsucker", "Caprimulgus europaeus"],
     "eurnig1"),
    (["Sociable Lapwing", "Sociable Plover", "Vanellus gregarius", "Chettusia gregaria"],
     "soclap1"),
    (["Demoiselle Crane", "Grus virgo", "Anthropoides virgo"], "demcra1"),
    (["Saker Falcon", "Falco cherrug"], "sakfal1"),
    (["Dunnock", "Hedge Sparrow", "Hedge Accentor"], "dunnoc1"),
])
def test_every_name_of_a_species_resolves_to_it(names_of_one_species, code):
    for name in names_of_one_species:
        assert resolve.resolve(f"What does the {name} eat?").species == (code,), name


@pytest.mark.parametrize("query, expected", [
    ("How swift is the Peregrine Falcon in a dive?", ("perfal",)),
    ("What little birds live in the steppe?", ()),
    ("Is a Hedge Sparrow a sparrow?", ("dunnoc1",)),
    ("What does the Bald Eagle eat?", ()),  # "eagle" is a group: no filter
    ("Where does the Black-necked Crane breed?", ()),
    ("What is the capital of Kazakhstan?", ()),
])
def test_committed_table_on_trap_questions(query, expected):
    assert resolve.resolve(query).species == expected


def test_committed_reviewed_ambiguous_aliases_ask():
    r = resolve.resolve("Is the cat owl nocturnal?")
    assert set(r.ambiguous.species) == {"loeowl", "tawowl1"}


def test_everyday_words_are_real_one_word_aliases_with_a_known_rule():
    """A typo or a word dropped from the alias table would make an entry do nothing."""
    everyday = pd.read_csv(resolve.EVERYDAY_WORDS_PATH)
    assert everyday["word"].is_unique
    assert set(everyday["rule"]) <= set(resolve.RULES)
    index = resolve.default_index()
    missing = [w for w in everyday["word"] if w not in index.entries or " " in w]
    assert not missing, missing


def test_matches_record_where_the_question_says_the_name(names):
    query = "Is the Eurasian Eagle-Owl's diet like a Steppe Eagle's?"
    spans = [query[slice(*m.span)] for m in names.resolve(query).matches]
    assert spans == ["Eurasian Eagle-Owl's", "Steppe Eagle's"]
    query = "What do  COMMON goatsuckers eat?"
    (match,) = names.resolve(query).matches
    assert query[slice(*match.span)] == "COMMON goatsuckers"


def test_choose_keeps_the_span(names):
    (cat_owl,) = names.resolve("is the cat owl nocturnal").matches
    assert cat_owl.choose("tawowl1").span == cat_owl.span == (7, 14)
