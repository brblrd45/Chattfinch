# Data sources and licenses

What each source contributes, how it's stored, and what the terms mean for this
repository. **Status: local only, not yet published.** Items marked ⚠ are open and must
be settled before the repository or any image is made public.

## Summary

| Source | Used for | Where it lives | Redistribution |
|---|---|---|---|
| eBird API | Species list, taxonomy, common names; live recent sightings | `data/dumps/species_list.csv`, `aliases.csv`, `corpus.jsonl.gz` (profile sections) | ⚠ Terms not yet checked |
| GBIF API | Name matching, English vernacular names | `aliases.csv`, `taxonomy_mismatches.csv` | ⚠ Terms not yet checked |
| Wikipedia | Article text, article titles and redirects | `corpus.jsonl.gz`, `aliases.csv` | CC BY-SA 4.0, with conditions below ⚠ |
| IUCN Red List API | Conservation status; English common names | Status: `data/private/` only. Names: `aliases.csv` | **Status data: not redistributable** |

## Wikipedia (CC BY-SA 4.0)

Wikipedia text is licensed CC BY-SA 4.0. `corpus.jsonl.gz` contains that text, split into
sections, and the vector store contains it split into chunks. Both are adaptations, so:

- **Attribution.** Every corpus document records the article title, the **permalink of
  the exact revision** the text came from, its timestamp, the retrieval date and the
  license (`wikipedia` field), and every section and chunk carries that permalink as its
  `url`. The app must show the source link next to anything it quotes from Wikipedia.
- **ShareAlike.** The corpus (and anything that republishes its text) must itself be
  shared under CC BY-SA 4.0 or a compatible license. This is a different license from the
  one chosen for this project's code; keep the corpus clearly separated and labeled.
- Decision (project owner, 2026-10-15): only the source code is shared; the corpus and
  other data files are not published. The bot quotes Wikipedia text to its users with
  the attribution above.

## IUCN Red List (no redistribution)

IUCN's terms do not allow redistributing its data.

- The status snapshot is written to `data/private/`, which is git-ignored and
  docker-ignored; a pre-commit hook blocks committing anything under it. A fresh clone
  regenerates it with its own IUCN API token (`scripts/build_dumps.py`).
- IUCN status is deliberately **not** in the corpus (committed) or the vector store
  (built from the corpus, so it holds none either); the app reads it from the private
  snapshot at query time.
- Test fixtures for IUCN are synthetic. A recorded response was removed from the git
  history before any publication.
- **Decision (project owner): IUCN English common names are kept as one alias source** in
  `aliases.csv`. Of 3,702 alias rows, 5 rest on IUCN alone; the rest are also given by
  eBird, GBIF or Wikipedia. If this is ever judged to be IUCN data, drop `iucn` from
  `aliases.raw_names` and rebuild; the cost is those 5 names.
- **The bot shows IUCN categories to its users** (day 4 status answers: category,
  assessment year, link to the assessment, "Red List data retrieved <date>"). Today the
  bot is private (an allowlist of user IDs), which is personal, non-commercial use.
  ⚠ **Before opening the bot to the public, get a definite answer from IUCN.**
  Checked 2026-10-18 without a definite result: the official terms page
  (iucnredlist.org/terms/terms-of-use) refused automated access (HTTP 403), and the
  secondary sources found quote the terms as forbidding reposting or redistributing Red
  List data "in whole, or in part, by any means", including via the API, without written
  permission. Showing a category to members of the public could fall under that. Read
  the terms in a browser, or ask the Red List Unit.

## eBird and GBIF

Live sightings (day 4): when asked "seen recently?", the bot fetches eBird's recent
observations for one species and shows up to five (date, place name, count), credited
"eBird recent sightings" with a link to the species' eBird page for the region. Places
eBird flags as private (personal locations) are shown as "a private location". Nothing
is stored.

⚠ Not yet reviewed. Before publishing, check: the eBird API terms of use (the API key is
personal; redistributing API-derived data may be restricted) and the license of the
eBird/Clements taxonomy; and the GBIF backbone and vernacular-name licenses (GBIF
vernacular names are aggregated from other datasets with their own licenses).
The dumps contain names and taxonomy only, not observation data. Live sightings are
fetched at query time and never stored.

## Regenerating

Every file in `data/dumps/` is produced by `scripts/build_dumps.py` from the live APIs,
and the vector store by `scripts/build_index.py`, so a recipient can rebuild
everything from the sources under their own keys and terms instead of relying on the
committed copies.
