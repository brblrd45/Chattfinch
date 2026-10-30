import pandas as pd

from chattfinch import iucn_snapshot

SPECIES = pd.DataFrame({
    "species_code": ["aaa1", "bbb1", "ccc1", "ddd1"],
    "common_name": ["A Bird", "B Bird", "C Bird", "D Bird"],
    "scientific_name": ["Genus a", "Genus b", "Newgenus c", "Genus d"],
})

STATUS = {"scientific_name": "Genus a", "category": "VU", "year_published": "2021",
          "assessment_url": "https://www.iucnredlist.org/species/1/2", "english_names": []}

RECORDS = {
    # Found directly; everything agrees.
    "aaa1": {"gbif": {"match": {"accepted_name": "Genus a", "canonical_name": "Genus a",
                                "status": "ACCEPTED"}, "vernacular_names": []},
             "iucn": {"queried_name": "Genus a", "status": STATUS},
             "wikipedia": {"article": {"title": "A bird", "section": None, "redirects": []}},
             "iucn_fetched_on": "2026-10-07"},
    # GBIF calls it something else; IUCN only knows it under GBIF's name.
    "bbb1": {"gbif": {"match": {"accepted_name": "Oldgenus b", "canonical_name": "Genus b",
                                "status": "SYNONYM"}, "vernacular_names": []},
             "iucn": {"queried_name": "Oldgenus b", "status": {**STATUS, "category": "LC"}},
             "wikipedia": {"article": {"title": "Oldgenus", "section": "Species",
                                       "redirects": []}},
             "iucn_fetched_on": "2026-10-08"},
    # Unknown to GBIF and IUCN; no Wikipedia article.
    "ccc1": {"gbif": {"match": None, "vernacular_names": []},
             "iucn": {"queried_name": None, "status": None},
             "wikipedia": {"article": None},
             "iucn_fetched_on": "2026-10-08"},
    # Not fetched yet.
    "ddd1": {},
}


def test_snapshot_entries_and_fetch_dates():
    snapshot = iucn_snapshot.build_snapshot(SPECIES, RECORDS)
    assert snapshot["fetched"] == {"from": "2026-10-07", "to": "2026-10-08"}
    assert "not for redistribution" in snapshot["terms"]
    entries = snapshot["species"]
    assert set(entries) == {"aaa1", "bbb1", "ccc1"}  # ddd1 has no IUCN record yet
    assert entries["aaa1"]["category"] == "VU"
    assert entries["bbb1"] == {"scientific_name": "Genus b", "iucn_name": "Oldgenus b",
                               "category": "LC", "year_published": "2021",
                               "assessment_url": STATUS["assessment_url"]}
    assert entries["ccc1"]["category"] is None and entries["ccc1"]["iucn_name"] is None


def test_summarize_counts_categories():
    snapshot = iucn_snapshot.build_snapshot(SPECIES, RECORDS)
    assert iucn_snapshot.summarize(snapshot) == {"VU": 1, "LC": 1, "none": 1}


def test_mismatches():
    table = iucn_snapshot.build_mismatches(SPECIES, RECORDS)
    assert list(table.columns) == iucn_snapshot.MISMATCH_COLUMNS
    found = set(zip(table["species_code"], table["source"], table["issue"]))
    assert found == {
        ("bbb1", "gbif", "other_accepted_name"),
        ("bbb1", "iucn", "found_under_other_name"),
        ("bbb1", "wikipedia", "section_of_other_article"),
        ("ccc1", "gbif", "no_match"),
        ("ccc1", "iucn", "not_found"),
        ("ccc1", "wikipedia", "no_article"),
    }
    row = table[(table["species_code"] == "bbb1") & (table["source"] == "wikipedia")].iloc[0]
    assert row["source_name"] == "Oldgenus#Species"


def test_mismatches_flags_doubtful_gbif_match():
    records = {"aaa1": {"gbif": {"match": {"accepted_name": "Genus a",
                                           "canonical_name": "Genus a", "status": "DOUBTFUL"},
                                 "vernacular_names": []}}}
    table = iucn_snapshot.build_mismatches(SPECIES, records)
    assert list(table["issue"]) == ["doubtful"]
