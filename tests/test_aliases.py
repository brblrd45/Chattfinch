import pandas as pd
import pytest

from chattfinch.aliases import (alias_index, ambiguous_aliases, build_alias_table,
                              clean_candidates, head_noun, normalize)


@pytest.mark.parametrize("raw, expected", [
    ("European Nightjar", "european nightjar"),
    ("  Common   goatsucker ", "common goatsucker"),
    ("Black-winged Stilt", "black winged stilt"),
    ("Pallas's Sandgrouse", "pallas sandgrouse"),
    ("Pallas’s Sandgrouse", "pallas sandgrouse"),  # curly apostrophe
    ("Pallas Sandgrouse", "pallas sandgrouse"),
    ("Fern-owl", "fern owl"),
    ("Greylag Geese", "greylag goose"),
    ("barn owls", "barn owl"),
    ("Glossy Ibis", "glossy ibis"),  # not a plural
    ("Common Rosefinch.", "common rosefinch"),
    ("Hümé's Owl", "hume owl"),  # diacritics
    ("nightjar, european nightjar", "nightjar european nightjar"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_normalize_is_idempotent():
    for name in ["Pallas's Sandgrouse", "Greylag Geese", "Black-winged Stilts"]:
        assert normalize(normalize(name)) == normalize(name)


def test_head_noun():
    assert head_noun("Eurasian Curlew") == "curlew"
    assert head_noun("Tundra Bean-Goose") == "goose"
    assert head_noun("Brambling") == "brambling"


@pytest.mark.parametrize("raw, expected", [
    ("Common Goatsucker", ["Common Goatsucker"]),
    ("Cuckoo, Common Cuckoo", ["Cuckoo", "Common Cuckoo"]),
    ("Rock Pigeon (Feral Pigeon)", ["Feral Pigeon", "Rock Pigeon"]),
    ("Ruff (bird)", ["Ruff"]),
    ("Greater/Lesser Scaup", []),
    ("Long-legged x Upland Buzzard (hybrid)", []),
    ("EUJA", []),
    ("  ", []),
    ("Ox", []),  # too short to be a useful alias
    ("Chlidonias hybrida", ["Chlidonias hybrida"]),  # "hybrid" only as a whole word
])
def test_clean_candidates(raw, expected):
    assert clean_candidates(raw) == expected


SPECIES = pd.DataFrame({
    "species_code": ["eurcur", "whimbr", "eurnig1", "egynig1", "bkbplo", "ashou1"],
    "common_name": ["Eurasian Curlew", "Eurasian Whimbrel", "Eurasian Nightjar",
                    "Egyptian Nightjar", "Black-bellied Plover", "Asian Houbara"],
    "scientific_name": ["Numenius arquata", "Numenius phaeopus", "Caprimulgus europaeus",
                        "Caprimulgus aegyptius", "Pluvialis squatarola", "Chlamydotis macqueenii"],
})


def record(gbif=(), iucn=(), wikipedia=None, section=None, gbif_accepted=None):
    rec = {"gbif": {"match": {"accepted_name": gbif_accepted} if gbif_accepted else None,
                    "vernacular_names": list(gbif)},
           "iucn": {"status": None}}
    if iucn:
        rec["iucn"]["status"] = {"scientific_name": "X y", "english_names": list(iucn)}
    if wikipedia:
        rec["wikipedia"] = {"article": {"title": wikipedia[0], "redirects": list(wikipedia[1:]),
                                        "section": section}}
    return rec


def build(records, overrides=None):
    return build_alias_table(SPECIES, records, overrides)


def pairs(table):
    return set(zip(table["normalized"], table["species_code"]))


def test_ebird_and_scientific_names_always_included():
    table = build({})
    assert ("eurasian nightjar", "eurnig1") in pairs(table)
    assert ("caprimulgus europaeus", "eurnig1") in pairs(table)
    row = table[table["normalized"] == "eurasian nightjar"].iloc[0]
    assert row["alias"] == "Eurasian Nightjar" and row["sources"] == "ebird"


def test_goatsucker_and_wikipedia_redirects():
    table = build({"eurnig1": record(gbif=["Goatsucker"], wikipedia=[
        "European nightjar", "Common Nightjar", "Common goatsucker"])})
    index = alias_index(table)
    for name in ["Common goatsucker", "common nightjar", "European Nightjar", "goatsucker"]:
        assert index[normalize(name)] == ["eurnig1"]


def test_redirect_into_section_is_not_used():
    table = build({"eurnig1": record(wikipedia=["Caprimulgus", "Some other nightjar"],
                                     section="Species")})
    assert "some other nightjar" not in set(table["normalized"])


def test_shared_head_noun_becomes_group_alias():
    # "Nightjar" listed by GBIF for one species maps to every nightjar, as a group.
    table = build({"eurnig1": record(gbif=["Nightjar"])})
    rows = table[table["normalized"] == "nightjar"]
    assert set(rows["species_code"]) == {"eurnig1", "egynig1"}
    assert set(rows["kind"]) == {"group"}


def test_ebird_name_belongs_only_to_its_species():
    table = build({"whimbr": record(gbif=["Eurasian Curlew"])})
    assert alias_index(table)["eurasian curlew"] == ["eurcur"]


def test_ebird_scientific_name_belongs_only_to_its_species():
    table = build({"eurcur": record(wikipedia=["Eurasian curlew", "Numenius phaeopus"])})
    assert alias_index(table)[normalize("Numenius phaeopus")] == ["whimbr"]


def test_gbif_only_pairing_dropped_when_confirmed_elsewhere():
    table = build({"bkbplo": record(gbif=["Grey Plover"], iucn=["Grey Plover"]),
                   "ashou1": record(gbif=["Grey Plover"])})
    assert alias_index(table)["grey plover"] == ["bkbplo"]


def test_gbif_only_conflict_stays_ambiguous():
    table = build({"eurcur": record(gbif=["Curlew"]), "whimbr": record(gbif=["Curlew"])})
    assert ambiguous_aliases(alias_index(table))["curlew"] == ["eurcur", "whimbr"]


def test_gbif_accepted_name_added_as_scientific_alias():
    table = build({"eurnig1": record(gbif_accepted="Caprimulgus europaeus meridionalis")})
    row = table[table["normalized"] == normalize("Caprimulgus europaeus meridionalis")]
    assert row.iloc[0]["sources"] == "scientific"


def test_manual_overrides():
    overrides = pd.DataFrame({"alias": ["Goatsucker", "Dor hawk"],
                              "species_code": ["eurnig1", "eurnig1"],
                              "action": ["remove", "add"], "note": ["", ""]})
    table = build({"eurnig1": record(gbif=["Goatsucker"])}, overrides)
    assert ("goatsucker", "eurnig1") not in pairs(table)
    assert table[table["normalized"] == "dor hawk"].iloc[0]["sources"] == "manual"


def test_unknown_override_action_raises():
    overrides = pd.DataFrame({"alias": ["X bird"], "species_code": ["eurnig1"],
                              "action": ["rename"], "note": [""]})
    with pytest.raises(ValueError, match="rename"):
        build({}, overrides)


def test_ambiguous_aliases():
    index = {"buzzard": ["combuz1", "lolbuz1"], "eurasian nightjar": ["eurnig1"]}
    assert ambiguous_aliases(index) == {"buzzard": ["combuz1", "lolbuz1"]}
