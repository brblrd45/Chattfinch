"""IUCN Red List status snapshot, and the taxonomy-mismatch log.

Both are built from the records cached by name_sources.py; nothing is fetched here.

The snapshot holds IUCN data, which IUCN's terms don't allow redistributing, so it
goes to data/private/ (never committed). The mismatch log holds only names and
issue types, so it goes to data/dumps/ with the other dumps.
"""

import pandas as pd

MISMATCH_COLUMNS = ["species_code", "common_name", "scientific_name", "source",
                    "source_name", "issue"]


def build_snapshot(species: pd.DataFrame, records: dict[str, dict]) -> dict:
    """{"source", "fetched": {"from", "to"}, "species": {code: entry}} with one
    entry per species that has a cached IUCN record. An entry's `category` is None
    if IUCN has no global assessment under eBird's or GBIF's name."""
    entries, dates = {}, set()
    for row in species.itertuples():
        record = records.get(row.species_code, {})
        if "iucn" not in record:
            continue
        dates.add(record.get("iucn_fetched_on"))
        status = record["iucn"]["status"] or {}
        entries[row.species_code] = {
            "scientific_name": row.scientific_name,
            "iucn_name": record["iucn"]["queried_name"],
            "category": status.get("category"),
            "year_published": status.get("year_published"),
            "assessment_url": status.get("assessment_url"),
        }
    dates.discard(None)
    return {
        "source": "IUCN Red List API v4, latest global assessment per species",
        "terms": "IUCN data: not for redistribution. Keep out of public repositories.",
        "fetched": {"from": min(dates, default=None), "to": max(dates, default=None)},
        "species": entries,
    }


def build_mismatches(species: pd.DataFrame, records: dict[str, dict]) -> pd.DataFrame:
    """One row per place another source disagrees with eBird's taxonomy, for review.

    Issues: gbif no_match / other_accepted_name / doubtful; iucn not_found /
    found_under_other_name; wikipedia no_article / section_of_other_article.
    """
    rows = []

    def add(row, source, source_name, issue):
        rows.append({"species_code": row.species_code, "common_name": row.common_name,
                     "scientific_name": row.scientific_name, "source": source,
                     "source_name": source_name, "issue": issue})

    for row in species.itertuples():
        record = records.get(row.species_code, {})
        if "gbif" in record:
            match = record["gbif"]["match"]
            if match is None:
                add(row, "gbif", None, "no_match")
            else:
                if match["accepted_name"] != row.scientific_name:
                    add(row, "gbif", match["accepted_name"], "other_accepted_name")
                if match["status"] == "DOUBTFUL":
                    add(row, "gbif", match["canonical_name"], "doubtful")
        if "iucn" in record:
            iucn_name = record["iucn"]["queried_name"]
            if iucn_name is None:
                add(row, "iucn", None, "not_found")
            elif iucn_name != row.scientific_name:
                add(row, "iucn", iucn_name, "found_under_other_name")
        if "wikipedia" in record:
            article = record["wikipedia"]["article"]
            if article is None:
                add(row, "wikipedia", None, "no_article")
            elif article["section"]:
                add(row, "wikipedia", f"{article['title']}#{article['section']}",
                    "section_of_other_article")
    return pd.DataFrame(rows, columns=MISMATCH_COLUMNS)


def summarize(snapshot: dict) -> dict[str, int]:
    """Count of species per Red List category ("none" = no global assessment found)."""
    counts = pd.Series([e["category"] or "none" for e in snapshot["species"].values()])
    return counts.value_counts().to_dict()
