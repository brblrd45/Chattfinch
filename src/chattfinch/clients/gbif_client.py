"""GBIF API client. No key needed."""

from chattfinch.clients._http import ApiClient

BASE_URL = "https://api.gbif.org/v1"
PAGE_SIZE = 300

_client: ApiClient | None = None


def _get_client() -> ApiClient:
    global _client
    if _client is None:
        _client = ApiClient(BASE_URL, min_interval_s=0.2)
    return _client


def match_name(scientific_name: str) -> dict | None:
    """Resolve a bird's scientific name against the GBIF backbone (GET /species/match).

    Returns None unless the match is at species rank: for an unknown name GBIF
    returns a HIGHERRANK match to a genus or to Aves itself, not an empty result.
    `accepted_key` is the key to use for further lookups; it differs from
    `usage_key` when the name is a synonym (e.g. Grus virgo -> Anthropoides virgo).
    """
    data = _get_client().get_json("/species/match",
                                  params={"name": scientific_name, "class": "Aves"})
    if data.get("matchType") in (None, "NONE", "HIGHERRANK") or data.get("rank") != "SPECIES":
        return None
    return {
        "usage_key": data["usageKey"],
        "accepted_key": data.get("acceptedUsageKey", data["usageKey"]),
        "canonical_name": data["canonicalName"],
        "accepted_name": data.get("species", data["canonicalName"]),
        "status": data["status"],
        "match_type": data["matchType"],
        "confidence": data["confidence"],
    }


def vernacular_names(usage_key: int, language: str = "eng") -> list[str]:
    """Common names in one language for a taxon (GET /species/{key}/vernacularNames),
    exact duplicates removed, in first-seen order. Raw: includes generic names
    ("Nightjar") and odd formats ("nightjar, european nightjar") that aliases.py
    has to filter."""
    names: list[str] = []
    offset = 0
    while True:
        data = _get_client().get_json(f"/species/{usage_key}/vernacularNames",
                                      params={"limit": PAGE_SIZE, "offset": offset})
        for record in data["results"]:
            name = record.get("vernacularName", "").strip()
            if record.get("language") == language and name and name not in names:
                names.append(name)
        if data.get("endOfRecords", True):
            return names
        offset += PAGE_SIZE
