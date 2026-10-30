"""Per-species corpus documents with source-tagged sections (corpus.jsonl.gz).

Each document has:
  - an "ebird" Profile section generated from the taxonomy and the species'
    aliases, so retrieval finds the species under any of its names;
  - the Wikipedia article's sections, each citing the exact revision it came from.

IUCN status is deliberately not in the corpus: the corpus is committed, and IUCN
data can't be redistributed. The app reads status from the private snapshot.
"""

import gzip
import json
from pathlib import Path

import pandas as pd

from chattfinch import config

WIKIPEDIA_LICENSE = "CC BY-SA 4.0"


def ebird_url(species_code: str) -> str:
    return f"https://ebird.org/species/{species_code}"


def species_aliases(alias_table: pd.DataFrame, species_code: str,
                    exclude: tuple[str, ...] = ()) -> list[str]:
    """Display forms of a species' own aliases (no group names), one per
    normalized form, alphabetical, minus `exclude`."""
    rows = alias_table[(alias_table["species_code"] == species_code)
                       & (alias_table["kind"] == "species")]
    excluded = {name.lower() for name in exclude}
    return sorted({alias for alias in rows["alias"] if alias.lower() not in excluded},
                  key=str.lower)


def profile_text(row, aliases: list[str]) -> str:
    text = (f"{row.common_name} ({row.scientific_name}) is a bird species in the family "
            f"{row.family} ({row.family_common_name}), order {row.order}. It is on the eBird "
            f"list of species recorded in {config.REGION_NAME}.")
    if aliases:
        text += " Also known as: " + "; ".join(aliases) + "."
    return text


def build_document(row, record: dict, alias_table: pd.DataFrame) -> dict:
    aliases = species_aliases(alias_table, row.species_code,
                              exclude=(row.common_name, row.scientific_name))
    sections = [{"source": "ebird", "section": "Profile",
                 "text": profile_text(row, aliases), "url": ebird_url(row.species_code)}]
    article = record.get("article")
    wikipedia = None
    if article:
        wikipedia = {"title": article["title"], "revision_id": article["revision_id"],
                     "revision_timestamp": article["revision_timestamp"], "url": article["url"],
                     "retrieved_on": record.get("article_fetched_on"),
                     "license": WIKIPEDIA_LICENSE}
        sections += [{"source": "wikipedia", "section": name, "text": text, "url": article["url"]}
                     for name, text in article["sections"].items()]
    return {
        "species_code": row.species_code,
        "common_name": row.common_name,
        "scientific_name": row.scientific_name,
        "family": row.family,
        "family_common_name": row.family_common_name,
        "order": row.order,
        "aliases": aliases,
        "wikipedia": wikipedia,
        "sections": sections,
    }


def build_corpus(species: pd.DataFrame, records: dict[str, dict],
                 alias_table: pd.DataFrame) -> list[dict]:
    """One document per species, in species-list order. A species without a cached
    Wikipedia article still gets a document (the eBird profile only); use
    coverage_misses() to list those."""
    return [build_document(row, records.get(row.species_code, {}), alias_table)
            for row in species.itertuples()]


def coverage_misses(documents: list[dict]) -> list[str]:
    """Species codes whose document has no Wikipedia text."""
    return [d["species_code"] for d in documents
            if not any(s["source"] == "wikipedia" for s in d["sections"])]


def write_corpus(documents: list[dict], path: Path) -> None:
    """JSON lines, gzipped with no timestamp or filename in the header, so an
    unchanged corpus rebuilds to byte-identical output (no spurious git diffs)."""
    with open(path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
        for document in documents:
            gz.write((json.dumps(document, ensure_ascii=False) + "\n").encode("utf-8"))


def read_corpus(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
