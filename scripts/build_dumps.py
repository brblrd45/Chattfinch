"""Regenerate the published dumps in data/dumps/ from the source APIs.

Run from the repo root:  .venv/bin/python scripts/build_dumps.py

Implemented:
  species_list.csv   the target species and their eBird/Clements taxonomy
                     (this is also the plan's "taxonomy table"). FROZEN: read from
                     the committed file; re-pull from eBird only with
                     --refresh-species-list
  aliases.csv        every alias with its species, kind (species/group) and sources,
                     for review and for display ("also known as")
  alias_index.json   {normalized alias: [species codes]}, used at query time
  taxonomy_mismatches.csv  where GBIF/IUCN/Wikipedia disagree with eBird's taxonomy
  corpus.jsonl.gz    one document per species: an eBird profile plus the Wikipedia
                     article's sections (CC BY-SA 4.0, each citing its revision)
  manifest.json      build date, region and settings, row counts

Also written, but to data/private/ (never committed, IUCN terms forbid
redistribution):
  iucn_snapshot.json  latest global Red List category per species, with fetch dates

Raw API responses are cached in data/raw/name_sources/ (not committed); delete
that directory to refetch them.

"""

import argparse
import json
from datetime import date

import pandas as pd

from chattfinch import aliases, config, corpus, iucn_snapshot, name_sources, species_list


def build_species_list(refresh: bool) -> pd.DataFrame:
    """The frozen species list from data/dumps/species_list.csv, unless `refresh`
    re-pulls it from eBird (the only time this script reads the eBird list)."""
    notable = species_list.load_notable()
    if refresh:
        species, changes = species_list.refresh(config.REGION_CODE, notable, config.FREQUENCY_FILE,
                                                config.TARGET_SPECIES_COUNT)
        if changes is None:
            print("species_list.csv: created from eBird (there was no frozen list)")
        elif any(changes.values()):
            print("species_list.csv: REFRESHED from eBird; changes since the frozen list:")
            print("\n".join(species_list.describe_changes(changes)))
            print("  Review with git diff, then commit the new list to freeze it.")
        else:
            print("species_list.csv: refreshed from eBird; no changes")
    else:
        species = species_list.load_frozen(notable)
    print(f"species_list.csv: {len(species)} species, {species['is_notable'].sum()} notable, "
          f"{species['family'].nunique()} families" + ("" if refresh else " (frozen)"))
    return species


def collect_name_sources(species: pd.DataFrame) -> dict[str, dict]:
    sources = name_sources.SOURCES
    if "example.com" in config.WIKIPEDIA_USER_AGENT:
        print("WARNING: WIKIPEDIA_USER_AGENT in .env is still the placeholder; Wikipedia "
              "throttles it, so Wikipedia names and articles are skipped this run. Set a real contact "
              "address and rerun to add them.")
        sources = tuple(s for s in sources if s not in ("wikipedia", "article"))
    records = name_sources.collect(species, sources)
    for source in name_sources.SOURCES:
        missing = sum(source not in record for record in records.values())
        if missing:
            print(f"  {source}: no data yet for {missing} species")
    return records


def build_aliases(species: pd.DataFrame, records: dict[str, dict]) -> pd.DataFrame:
    table = aliases.build_alias_table(species, records, aliases.load_overrides())
    index = aliases.alias_index(table)
    table.to_csv(config.DUMPS_DIR / "aliases.csv", index=False)
    (config.DUMPS_DIR / "alias_index.json").write_text(
        json.dumps(index, indent=0, ensure_ascii=False) + "\n")
    ambiguous = aliases.ambiguous_aliases(index)
    groups = set(table.loc[table["kind"] == "group", "normalized"])
    print(f"aliases.csv: {len(table)} rows, {len(index)} distinct aliases; ambiguous: "
          f"{len(set(ambiguous) & groups)} group names, {len(set(ambiguous) - groups)} other")
    return table


def build_iucn_snapshot(species: pd.DataFrame, records: dict[str, dict]) -> dict:
    snapshot = iucn_snapshot.build_snapshot(species, records)
    config.PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    (config.PRIVATE_DIR / "iucn_snapshot.json").write_text(
        json.dumps(snapshot, indent=1, ensure_ascii=False) + "\n")
    print(f"private/iucn_snapshot.json: {len(snapshot['species'])} species, fetched "
          f"{snapshot['fetched']['from']}..{snapshot['fetched']['to']}; "
          f"categories {iucn_snapshot.summarize(snapshot)}")
    return snapshot


def build_mismatches(species: pd.DataFrame, records: dict[str, dict]) -> pd.DataFrame:
    mismatches = iucn_snapshot.build_mismatches(species, records)
    mismatches.to_csv(config.DUMPS_DIR / "taxonomy_mismatches.csv", index=False)
    counts = mismatches.groupby(["source", "issue"]).size().to_dict()
    print(f"taxonomy_mismatches.csv: {len(mismatches)} rows {counts}")
    return mismatches


def build_corpus(species: pd.DataFrame, records: dict[str, dict],
                 alias_table: pd.DataFrame) -> list[dict]:
    documents = corpus.build_corpus(species, records, alias_table)
    corpus.write_corpus(documents, config.DUMPS_DIR / "corpus.jsonl.gz")
    sections = [s for d in documents for s in d["sections"] if s["source"] == "wikipedia"]
    misses = corpus.coverage_misses(documents)
    print(f"corpus.jsonl.gz: {len(documents)} documents, {len(sections)} Wikipedia sections, "
          f"{sum(len(s['text']) for s in sections) / 1e6:.1f}M characters; "
          f"{len(misses)} species without Wikipedia text{': ' + ', '.join(misses) if misses else ''}")
    return documents


def write_manifest(species: pd.DataFrame, alias_table: pd.DataFrame,
                   mismatches: pd.DataFrame, snapshot: dict, documents: list[dict]) -> None:
    manifest = {
        "built_on": date.today().isoformat(),
        "region": config.REGION_CODE,
        "frequency_file": str(config.FREQUENCY_FILE) if config.FREQUENCY_FILE else None,
        "files": {
            "species_list.csv": {"rows": len(species)},
            "aliases.csv": {"rows": len(alias_table),
                            "sources": sorted({s for v in alias_table["sources"] for s in v.split("|")})},
            "alias_index.json": {"aliases": alias_table["normalized"].nunique()},
            "taxonomy_mismatches.csv": {"rows": len(mismatches)},
            "corpus.jsonl.gz": {"documents": len(documents),
                                "missing_wikipedia": corpus.coverage_misses(documents),
                                "license": f"Wikipedia text: {corpus.WIKIPEDIA_LICENSE}"},
        },
        # Built alongside the dumps but kept out of the repo (IUCN terms); listed so
        # a fresh clone knows to regenerate it.
        "private_files": {
            "iucn_snapshot.json": {"species": len(snapshot["species"]),
                                   "fetched": snapshot["fetched"]},
        },
    }
    (config.DUMPS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def dumps_size_mb() -> float:
    return sum(f.stat().st_size for f in config.DUMPS_DIR.iterdir() if f.is_file()) / 1e6


def report_size() -> None:
    size = dumps_size_mb()
    status = "OVER BUDGET" if size > config.DUMPS_BUDGET_MB else "within budget"
    print(f"data/dumps/: {size:.1f} MB of {config.DUMPS_BUDGET_MB} MB budget ({status})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refresh-species-list", action="store_true",
                        help="re-pull the species list from eBird and report what changed "
                             "(by default the committed, frozen list is used)")
    args = parser.parse_args()
    config.DUMPS_DIR.mkdir(parents=True, exist_ok=True)
    species = build_species_list(args.refresh_species_list)
    records = collect_name_sources(species)
    alias_table = build_aliases(species, records)
    snapshot = build_iucn_snapshot(species, records)
    mismatches = build_mismatches(species, records)
    documents = build_corpus(species, records, alias_table)
    write_manifest(species, alias_table, mismatches, snapshot, documents)
    report_size()


if __name__ == "__main__":
    main()
