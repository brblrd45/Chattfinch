"""Name-alias index: {normalized alias -> [canonical species codes]}.

A list, not a single ID, because real common names collide ("curlew", "dipper").
Aliases with more than one candidate are ambiguous and the app asks the user
rather than guessing.

Built from the raw name records in name_sources.py:
  ebird       the eBird common name (authoritative)
  scientific  eBird's scientific name, plus GBIF/IUCN names when they differ
              (older genus placements like "Sylvia communis")
  gbif        GBIF vernacular names
  iucn        IUCN English names
  wikipedia   the species' article title and the titles redirecting to it
  manual      data/manual/alias_overrides.csv
"""

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

from chattfinch import config

_IRREGULAR_PLURALS = {"geese": "goose"}
# Singular words ending in s that a plural rule would otherwise mangle
# (pallas, ibis, grass, ...); plural bird names rarely end like this.
_NOT_PLURAL_ENDINGS = ("ss", "us", "is", "as", "os")
_BANDING_CODE = re.compile(r"^[A-Z]{2,6}$")
_HYBRID = re.compile(r"\bx\b|\bhybrid\b", re.IGNORECASE)
_PARENTHETICAL = re.compile(r"\(([^)]*)\)")

OVERRIDES_PATH = config.MANUAL_DIR / "alias_overrides.csv"
ALIAS_COLUMNS = ["normalized", "alias", "species_code", "kind", "sources"]


def normalize(name: str) -> str:
    """Canonical form of a bird name, used identically at build time and query time:
    lowercase, strip diacritics, drop possessive 's and other apostrophes, turn
    hyphens and punctuation into spaces, collapse whitespace, and fold simple
    plurals word by word ("Pallas's Sandgrouse" -> "pallas sandgrouse",
    "Black-winged Stilts" -> "black winged stilt")."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"'s\b", "", text).replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(_singular(word) for word in text.split())


def _singular(word: str) -> str:
    if word in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[word]
    if len(word) <= 3 or word.endswith(_NOT_PLURAL_ENDINGS) or not word.endswith("s"):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    return word[:-1]


def head_noun(name: str) -> str:
    """Last word of the normalized name: the group a bird belongs to
    ("Eurasian Curlew" -> "curlew", "Tundra Bean-Goose" -> "goose")."""
    return normalize(name).rsplit(" ", 1)[-1]


def clean_candidates(raw: str) -> list[str]:
    """Split one raw name into usable alias strings, or [] if it isn't a name of a
    single species.

    Drops slash names ("Greater/Lesser Scaup", eBird's either-or categories),
    hybrids and banding codes ("EUJA"). Splits comma lists ("Cuckoo, Common
    Cuckoo"). Keeps a multi-word parenthetical as a second name ("Rock Pigeon
    (Feral Pigeon)") but drops one-word disambiguators ("Ruff (bird)").
    """
    raw = raw.strip()
    if not raw or "/" in raw or _HYBRID.search(raw) or _BANDING_CODE.match(raw):
        return []
    pieces = []
    for inner in _PARENTHETICAL.findall(raw):
        if len(inner.split()) >= 2:
            pieces.append(inner)
    pieces += _PARENTHETICAL.sub(" ", raw).split(",")
    return [" ".join(p.split()) for p in pieces if len(normalize(p)) >= 3]


def raw_names(species_row, record: dict) -> list[tuple[str, str]]:
    """(raw name, source) pairs for one species from its name-source record."""
    names = [(species_row.common_name, "ebird"), (species_row.scientific_name, "scientific")]
    gbif = record.get("gbif") or {}
    if gbif.get("match"):
        names.append((gbif["match"]["accepted_name"], "scientific"))
    names += [(n, "gbif") for n in gbif.get("vernacular_names", [])]
    status = (record.get("iucn") or {}).get("status")
    if status:
        names.append((status["scientific_name"], "scientific"))
        names += [(n, "iucn") for n in status["english_names"]]
    article = (record.get("wikipedia") or {}).get("article")
    # A redirect into a section means the species has no article of its own; the
    # article's redirects belong to other taxa, so none of them are used.
    if article and not article["section"]:
        names += [(t, "wikipedia") for t in [article["title"], *article["redirects"]]]
    return names


def build_alias_table(species: pd.DataFrame, records: dict[str, dict],
                      overrides: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (normalized alias, species), columns ALIAS_COLUMNS.

    Rules, in order:
      1. Clean each raw name (clean_candidates).
      2. A one-word alias equal to a head noun shared by 2+ species ("curlew") is a
         group name, not a name for the species that happened to list it: it's
         replaced by a "group" alias mapping to every species with that head noun.
      3. An eBird common or scientific name belongs only to its own species: if
         another species' sources also list it, that pairing is dropped (e.g. a
         Wikipedia redirect "Buteo japonicus" -> Common Buzzard, left over from when
         Eastern Buzzard was a subspecies).
      4. GBIF vernacular names are the noisiest source. If an alias is confirmed by
         another source for some species, pairings of it with other species that
         rest on GBIF alone are dropped ("grey plover" is IUCN-backed for
         Black-bellied Plover; GBIF also lists it for Asian Houbara).
      5. Manual overrides are applied last (add / remove).
    """
    head_counts = species["common_name"].map(head_noun).value_counts()
    group_heads = set(head_counts[head_counts >= 2].index)
    ebird_owner = {normalize(name): code
                   for column in ("common_name", "scientific_name")
                   for code, name in zip(species["species_code"], species[column])}

    found: dict[tuple[str, str], dict] = {}  # (normalized, code) -> {"alias", "kind", "sources"}

    def add(alias: str, code: str, kind: str, source: str) -> None:
        entry = found.setdefault((normalize(alias), code),
                                 {"alias": alias, "kind": kind, "sources": set()})
        entry["sources"].add(source)

    for row in species.itertuples():
        for raw, source in raw_names(row, records.get(row.species_code, {})):
            for alias in clean_candidates(raw):
                key = normalize(alias)
                if key in group_heads and " " not in key:
                    continue  # handled as a group alias below
                if ebird_owner.get(key, row.species_code) != row.species_code:
                    continue
                add(alias, row.species_code, "species", source)

    confirmed = {key for (key, _), e in found.items() if e["sources"] - {"gbif"}}
    for (key, code), entry in list(found.items()):
        if entry["sources"] == {"gbif"} and key in confirmed:
            del found[(key, code)]

    for row in species.itertuples():
        head = head_noun(row.common_name)
        if head in group_heads:
            add(head.capitalize(), row.species_code, "group", "group")

    if overrides is not None:
        for row in overrides.itertuples():
            if row.action == "remove":
                found.pop((normalize(row.alias), row.species_code), None)
            elif row.action == "add":
                add(row.alias, row.species_code, "species", "manual")
            else:
                raise ValueError(f"Unknown override action {row.action!r} for {row.alias!r}")

    table = pd.DataFrame([{"normalized": key, "species_code": code, "alias": e["alias"],
                           "kind": e["kind"], "sources": "|".join(sorted(e["sources"]))}
                          for (key, code), e in found.items()], columns=ALIAS_COLUMNS)
    return table.sort_values(["normalized", "species_code"], ignore_index=True)


def alias_index(table: pd.DataFrame) -> dict[str, list[str]]:
    """{normalized alias -> sorted species codes} from an alias table."""
    return {key: sorted(group["species_code"]) for key, group in table.groupby("normalized")}


def load_overrides(path: Path = OVERRIDES_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def ambiguous_aliases(index: dict[str, list[str]]) -> dict[str, list[str]]:
    """Subset of the index whose aliases map to more than one species."""
    return {alias: codes for alias, codes in index.items() if len(codes) > 1}
