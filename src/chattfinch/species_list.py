"""The fixed target species list (data/dumps/species_list.csv).

The list is **frozen**: the committed CSV is the source of truth and the build reads
it without calling eBird (`load_frozen`). It changes only on an explicit refresh
(`refresh`), which re-pulls it from eBird and reports what was added, removed or
renamed, so a change is always visible and deliberate.

A refresh starts from the eBird list for the region (category "species" only, which
drops hybrids, slashes and spuhs) and flags data/manual/notable_species.csv entries.
Ranking by reporting frequency is optional and off by default (see
config.FREQUENCY_FILE). Canonical ID = eBird species code.
"""

from pathlib import Path

import pandas as pd

from chattfinch import config
from chattfinch.clients import ebird_client

COLUMNS = ["species_code", "common_name", "scientific_name", "order", "family",
           "family_common_name", "taxon_order", "is_notable", "mean_frequency"]


def load_notable(path: Path = config.MANUAL_DIR / "notable_species.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_frequencies(path: Path) -> pd.DataFrame:
    """Parse an eBird bar-chart export into columns: common_name, mean_frequency.
    Not implemented yet: frequency ranking is off for KZ, and the export format
    should be checked against a real download before writing the parser."""
    raise NotImplementedError("Frequency ranking isn't implemented; leave config.FREQUENCY_FILE as None")


def build_species_list(region_code: str, notable: pd.DataFrame,
                       frequency_file: Path | None = None,
                       target_count: int | None = None) -> pd.DataFrame:
    """One row per species, in taxonomic order (columns: COLUMNS).

    Without `frequency_file`, every species on the region list is kept. With it,
    only the top `target_count` by mean frequency are kept, plus every notable
    species. Raises if a notable species isn't on the region list, since that's
    usually a typo or a name the eBird taxonomy files differently.
    """
    taxa = ebird_client.taxonomy(ebird_client.region_species_codes(region_code))
    species = taxa[taxa["category"] == "species"].drop(columns="category")

    missing = set(notable["scientific_name"]) - set(species["scientific_name"])
    if missing:
        raise ValueError(f"Notable species not on the eBird {region_code} list "
                         f"(check spelling/taxonomy): {sorted(missing)}")
    species = species.assign(is_notable=species["scientific_name"].isin(notable["scientific_name"]),
                             mean_frequency=pd.NA)

    if frequency_file is not None:
        frequencies = load_frequencies(frequency_file)
        species = species.drop(columns="mean_frequency").merge(frequencies, on="common_name",
                                                               how="left")
        ranked = species.sort_values("mean_frequency", ascending=False)
        top_codes = set(ranked["species_code"].head(target_count))
        species = species[species["species_code"].isin(top_codes) | species["is_notable"]]

    return species[COLUMNS].reset_index(drop=True)


def load_frozen(notable: pd.DataFrame, path: Path = config.SPECIES_LIST_PATH) -> pd.DataFrame:
    """The committed species list, with `is_notable` recomputed from `notable` so
    editing notable_species.csv takes effect without a refresh. Raises
    FileNotFoundError if there is no frozen list yet, and ValueError if a notable
    species isn't on it (add it by refreshing, or fix the name)."""
    if not path.exists():
        raise FileNotFoundError(f"No frozen species list at {path}; create it with "
                                "scripts/build_dumps.py --refresh-species-list")
    species = pd.read_csv(path)
    missing = set(notable["scientific_name"]) - set(species["scientific_name"])
    if missing:
        raise ValueError(f"Notable species not on the frozen species list: {sorted(missing)}. "
                         "Fix the spelling, or refresh the list if eBird now includes them.")
    species["is_notable"] = species["scientific_name"].isin(notable["scientific_name"])
    return species[COLUMNS]


def diff(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """What a refresh changes: {"added": [(code, common name)], "removed": [(code,
    common name)], "changed": [(code, field, old value, new value)]}, the last for the
    identifying fields (names and family). Each list is empty if nothing changed."""
    old, new = old.set_index("species_code"), new.set_index("species_code")
    fields = ["common_name", "scientific_name", "family"]
    changed = [(code, field, old.at[code, field], new.at[code, field])
               for code in sorted(old.index.intersection(new.index)) for field in fields
               if old.at[code, field] != new.at[code, field]]
    return {"added": [(code, new.at[code, "common_name"]) for code in sorted(new.index.difference(old.index))],
            "removed": [(code, old.at[code, "common_name"]) for code in sorted(old.index.difference(new.index))],
            "changed": changed}


def refresh(region_code: str, notable: pd.DataFrame, frequency_file: Path | None = None,
            target_count: int | None = None,
            path: Path = config.SPECIES_LIST_PATH) -> tuple[pd.DataFrame, dict | None]:
    """Re-pull the list from eBird and write it to `path`. Returns the new list and
    its diff against the previous frozen list (None if there wasn't one)."""
    new = build_species_list(region_code, notable, frequency_file, target_count)
    changes = diff(pd.read_csv(path), new) if path.exists() else None
    new.to_csv(path, index=False)
    return new, changes


def describe_changes(changes: dict) -> list[str]:
    """Human-readable lines for a diff."""
    lines = [f"  added:   {code} {name}" for code, name in changes["added"]]
    lines += [f"  removed: {code} {name}" for code, name in changes["removed"]]
    lines += [f"  changed: {code} {field}: {before!r} -> {after!r}"
              for code, field, before, after in changes["changed"]]
    return lines
