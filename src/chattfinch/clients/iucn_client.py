"""IUCN Red List API v4 client, used only to take a one-off dated status snapshot.
Auth: `Authorization: Bearer <IUCN_API_TOKEN>`."""

import httpx

from chattfinch import config
from chattfinch.clients._http import ApiClient

BASE_URL = "https://api.iucnredlist.org/api/v4"
GLOBAL_SCOPE = "1"

_client: ApiClient | None = None


def _get_client() -> ApiClient:
    global _client
    if _client is None:
        if not config.IUCN_API_TOKEN:
            raise RuntimeError("IUCN_API_TOKEN is not set; add it to .env")
        _client = ApiClient(BASE_URL, headers={"Authorization": f"Bearer {config.IUCN_API_TOKEN}"},
                            min_interval_s=0.5)
    return _client


def species_status(scientific_name: str) -> dict | None:
    """Latest *global* Red List assessment for a species, or None if IUCN doesn't
    know the name (404). IUCN can file a species under a different name than eBird
    (e.g. Anthropoides virgo, not Grus virgo), so callers should retry with the GBIF
    accepted name.

    The "latest" flag alone isn't enough: a species' newest assessment can be
    regional (e.g. Europe), so only Global-scope assessments are considered.
    `category` is None if there is no global assessment.
    """
    genus, _, species = scientific_name.strip().partition(" ")
    try:
        data = _get_client().get_json("/taxa/scientific_name",
                                      params={"genus_name": genus, "species_name": species})
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 404:
            return None
        raise
    global_assessments = [a for a in data["assessments"]
                          if any(s.get("code") == GLOBAL_SCOPE for s in a.get("scopes", []))]
    latest = next((a for a in global_assessments if a.get("latest")), None)
    return {
        "scientific_name": data["taxon"]["scientific_name"],
        "category": latest["red_list_category_code"] if latest else None,
        "year_published": latest["year_published"] if latest else None,
        "assessment_url": latest["url"] if latest else None,
        "english_names": _english_names(data["taxon"].get("common_names", [])),
    }


def _english_names(common_names: list[dict]) -> list[str]:
    """English common names, IUCN's main name first, exact duplicates removed."""
    english = sorted((c for c in common_names if c.get("language") == "eng"),
                     key=lambda c: not c.get("main"))
    return list(dict.fromkeys(c["name"] for c in english))
