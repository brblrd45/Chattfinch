"""Fetch and cache the raw source records for each species: names from GBIF, IUCN
and Wikipedia, and the Wikipedia article text for the corpus.

One file per species, data/raw/name_sources/<species_code>.json, holding one key
per source plus a `<source>_fetched_on` date. Sources are fetched independently:
a source that fails (network error, rate limit) is left out of the file and
retried on the next run, and finished sources are never refetched. Delete the
directory to refresh everything.

The IUCN record is reused for the status snapshot, so it's fetched here once.
The `article` source depends on `wikipedia` (it fetches the article that lookup
resolved to) and waits until that's cached.
"""

import json
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from chattfinch import config
from chattfinch.clients import gbif_client, iucn_client, wikipedia_client

CACHE_DIR = config.RAW_DIR / "name_sources"
SOURCES = ("gbif", "iucn", "wikipedia", "article")
MAX_CONSECUTIVE_FAILURES = 5  # then skip that source for the rest of the run


def fetch_gbif(scientific_name: str) -> dict:
    match = gbif_client.match_name(scientific_name)
    names = gbif_client.vernacular_names(match["accepted_key"]) if match else []
    return {"match": match, "vernacular_names": names}


def fetch_iucn(scientific_name: str, gbif_accepted_name: str | None) -> dict:
    """IUCN sometimes files a species under a different name than eBird; retry
    with GBIF's accepted name (e.g. Grus virgo -> Anthropoides virgo)."""
    for name in dict.fromkeys(filter(None, [scientific_name, gbif_accepted_name])):
        status = iucn_client.species_status(name)
        if status is not None:
            return {"queried_name": name, "status": status}
    return {"queried_name": None, "status": None}


def fetch_wikipedia(scientific_name: str, common_name: str) -> dict:
    """Look the article up by scientific name, falling back to the eBird name."""
    for query in (scientific_name, common_name):
        article = wikipedia_client.article_redirects(query)
        if article is not None:
            return {"queried_name": query, "article": article}
    return {"queried_name": None, "article": None}


def fetch_article(wikipedia_record: dict) -> dict | None:
    """Article text and revision for the article the `wikipedia` lookup found."""
    article = wikipedia_record["article"]
    return wikipedia_client.article_sections(article["title"]) if article else None


def collect(species: pd.DataFrame, sources: tuple[str, ...] = SOURCES,
            cache_dir: Path = CACHE_DIR) -> dict[str, dict]:
    """Raw name records for every species, keyed by species code, fetching only
    what isn't cached yet. Prints progress and a summary of failed fetches."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    records, failures = {}, []
    streaks = dict.fromkeys(sources, 0)
    for i, row in enumerate(species.itertuples(), 1):
        path = cache_dir / f"{row.species_code}.json"
        record = json.loads(path.read_text()) if path.exists() else {
            "species_code": row.species_code, "scientific_name": row.scientific_name}
        for source in sources:
            if source in record or streaks[source] >= MAX_CONSECUTIVE_FAILURES:
                continue
            if source == "article" and "wikipedia" not in record:
                continue  # needs the resolved title; fetched on a later run
            try:
                if source == "gbif":
                    record["gbif"] = fetch_gbif(row.scientific_name)
                elif source == "iucn":
                    gbif_match = (record.get("gbif") or {}).get("match") or {}
                    record["iucn"] = fetch_iucn(row.scientific_name, gbif_match.get("accepted_name"))
                elif source == "wikipedia":
                    record["wikipedia"] = fetch_wikipedia(row.scientific_name, row.common_name)
                elif source == "article":
                    record["article"] = fetch_article(record["wikipedia"])
            except httpx.HTTPError as error:
                failures.append((row.species_code, source, repr(error)[:120]))
                streaks[source] += 1
                if streaks[source] == MAX_CONSECUTIVE_FAILURES:
                    print(f"  {source}: {MAX_CONSECUTIVE_FAILURES} failures in a row, "
                          "skipping it for the rest of this run")
                continue
            streaks[source] = 0
            record[f"{source}_fetched_on"] = date.today().isoformat()
            path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
        records[row.species_code] = record
        if i % 50 == 0 or i == len(species):
            print(f"  name sources: {i}/{len(species)} species")
    if failures:
        print(f"  {len(failures)} fetches failed and will be retried next run, e.g.: {failures[:3]}")
    return records
