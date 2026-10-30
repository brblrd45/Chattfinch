"""eBird API 2.0 client. Auth: `X-eBirdApiToken` header with EBIRD_API_KEY."""

import pandas as pd

from chattfinch import config
from chattfinch.clients._http import ApiClient

BASE_URL = "https://api.ebird.org/v2"
TAXONOMY_BATCH = 200  # species codes per taxonomy request, to keep URLs short

_client: ApiClient | None = None
_live_client: ApiClient | None = None


def _headers() -> dict[str, str]:
    if not config.EBIRD_API_KEY:
        raise RuntimeError("EBIRD_API_KEY is not set; add it to .env")
    return {"X-eBirdApiToken": config.EBIRD_API_KEY}


def _get_client() -> ApiClient:
    """For batch jobs (build_dumps.py): patient retries."""
    global _client
    if _client is None:
        _client = ApiClient(BASE_URL, headers=_headers(), min_interval_s=0.2)
    return _client


def _get_live_client() -> ApiClient:
    """For the bot's query path: a short timeout and one quick retry, so a slow
    eBird costs the user seconds, not a minute."""
    global _live_client
    if _live_client is None:
        _live_client = ApiClient(BASE_URL, headers=_headers(), max_retries=1, backoff_s=0.5,
                                 timeout_s=config.LIVE_TIMEOUT_S)
    return _live_client


def region_species_codes(region_code: str) -> list[str]:
    """Every taxon code ever reported in a region (GET /product/spplist/{region}).
    Includes non-species taxa such as hybrids; filter on taxonomy category."""
    return _get_client().get_json(f"/product/spplist/{region_code}")


def taxonomy(species_codes: list[str]) -> pd.DataFrame:
    """eBird/Clements taxonomy rows for the given codes (GET /ref/taxonomy/ebird),
    in taxonomic order. Columns: species_code, common_name, scientific_name,
    category, order, family, family_common_name, taxon_order."""
    rows = []
    for start in range(0, len(species_codes), TAXONOMY_BATCH):
        batch = species_codes[start:start + TAXONOMY_BATCH]
        rows += _get_client().get_json("/ref/taxonomy/ebird",
                                       params={"fmt": "json", "species": ",".join(batch)})
    frame = pd.DataFrame([{
        "species_code": row["speciesCode"],
        "common_name": row["comName"],
        "scientific_name": row["sciName"],
        "category": row["category"],
        "order": row.get("order"),
        "family": row.get("familySciName"),
        "family_common_name": row.get("familyComName"),
        "taxon_order": row["taxonOrder"],
    } for row in rows], columns=["species_code", "common_name", "scientific_name", "category",
                                 "order", "family", "family_common_name", "taxon_order"])
    return frame.sort_values("taxon_order", ignore_index=True)


def recent_observations(region_code: str, species_code: str | None = None,
                        back_days: int = 14) -> list[dict]:
    """Recent sightings, newest first (GET /data/obs/{region}/recent[/{species}]).
    Live query path from day 4. eBird coverage of KZ is sparse, so an empty
    result means "no recent eBird reports", not "absent"."""
    path = f"/data/obs/{region_code}/recent" + (f"/{species_code}" if species_code else "")
    observations = _get_live_client().get_json(path, params={"back": back_days})
    return [{
        "species_code": obs["speciesCode"],
        "common_name": obs["comName"],
        "scientific_name": obs["sciName"],
        "location": obs["locName"],
        "location_private": obs.get("locationPrivate", False),
        "observed_at": obs["obsDt"],
        "count": obs.get("howMany"),
        "lat": obs["lat"],
        "lng": obs["lng"],
    } for obs in observations]
