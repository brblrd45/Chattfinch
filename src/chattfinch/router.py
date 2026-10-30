"""Query routing: send each question down the path that can answer it.

  descriptive  (default) retrieval over the Wikipedia corpus, answered by the model
  status       "Is the Saker Falcon endangered?": the Red List category, read from
               the private IUCN snapshot (data/private/iucn_snapshot.json)
  live         "Has the Sociable Lapwing been seen recently?": eBird's recent
               sightings in the region, fetched at question time

Status and live answers are written here from the data, without a model call, and
cite their source like any other answer, so the bot shows and links them the same way.

The classifier is a few regular expressions, deliberately narrow. When in doubt a
question stays descriptive, because the corpus can answer most things and the model
declines what it can't. In particular "why" questions about threats ("Why has the
Sociable Lapwing declined?") stay descriptive: the Red List category doesn't explain
anything, the Wikipedia article does.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from chattfinch import config
from chattfinch.resolve import Match

log = logging.getLogger("chattfinch.router")

DESCRIPTIVE, STATUS, LIVE = "descriptive", "status", "live"

# Decline reasons for the routed paths (the model's own are in prompts.py).
SPECIES_NEEDED = "species_needed"  # a status or live question that names no covered bird
STATUS_UNAVAILABLE = "status_unavailable"  # no IUCN snapshot on this machine
LIVE_UNAVAILABLE = "live_unavailable"  # eBird didn't answer

_LIVE = re.compile(
    r"\b(seen|spotted|sighted|reported|observed)\b.*\b(recently|lately|this (week|month|year)"
    r"|today|yesterday|now)\b"
    r"|\b(recently|lately)\b.*\b(seen|spotted|sighted|reported|observed)\b"
    r"|\b(recent|latest|last|most recent|any) (sightings?|reports?|observations?|records?)\b"
    r"|\bwhere can i (see|find)\b.*\b(now|currently|at the moment|this (week|month))\b")
_STATUS = re.compile(
    r"\b(conservation status|red[- ]?list|iucn)\b"
    r"|\bstatus of\b|\bwhat is (its|the) status\b"
    r"|\b(is|are)\b.*\b(endangered|threatened|vulnerable|extinct|at risk|in danger)\b"
    r"|\bhow (endangered|threatened)\b")
# Asking for a reason or for threats: the corpus answers that, not a category.
_EXPLAIN = re.compile(r"\b(why|how come|because|threats?|threatens|causes?|reasons?"
                      r"|declin\w*)\b")

CATEGORIES = {
    "LC": "Least Concern", "NT": "Near Threatened", "VU": "Vulnerable",
    "EN": "Endangered", "CR": "Critically Endangered", "EW": "Extinct in the Wild",
    "EX": "Extinct", "DD": "Data Deficient", "NE": "Not Evaluated",
}

NEED_SPECIES = {
    STATUS: ("I couldn't tell which bird you mean. I can give the conservation status of "
             f"birds recorded in {config.REGION_NAME}: name one, for example "
             "“Is the Saker Falcon endangered?”"),
    LIVE: ("I couldn't tell which bird you mean. I can check recent eBird sightings of "
           f"birds recorded in {config.REGION_NAME}: name one, for example "
           "“Has the Sociable Lapwing been seen recently?”"),
}
NO_SNAPSHOT = ("I can't look up conservation status right now: the IUCN Red List data "
               "isn't available on this server.")
NO_EBIRD = ("I couldn't reach eBird to check recent sightings just now. Please try again "
            "in a few minutes.")


def classify(query: str) -> str:
    """The route for `query`: LIVE, STATUS or DESCRIPTIVE."""
    text = query.lower()
    if _LIVE.search(text):
        return LIVE
    if _STATUS.search(text) and not _EXPLAIN.search(text):
        return STATUS
    return DESCRIPTIVE


@dataclass(frozen=True)
class Reply:
    """A routed answer, before rag wraps it: `sources` are rag.Source fields."""
    text: str
    declined: bool = False
    reason: str | None = None
    sources: list[dict] = field(default_factory=list)


def load_snapshot(path: Path = config.IUCN_SNAPSHOT_PATH) -> dict | None:
    """The IUCN snapshot, or None if this machine doesn't have it (a fresh clone
    without an IUCN token, or a container without data/private/ mounted). Read on
    every status question, so a snapshot mounted later is picked up."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def recent_sightings(species_code: str) -> list[dict]:
    """eBird's recent sightings of one species in the region (one per location,
    newest first). Raises httpx.HTTPError or RuntimeError (no API key) on failure."""
    from chattfinch.clients import ebird_client
    return ebird_client.recent_observations(config.REGION_CODE, species_code,
                                            back_days=config.RECENT_DAYS)


def _named_species(names: list[Match]) -> list[tuple[str, str, str]]:
    """(code, canonical name, scientific name) per species named, in order,
    without repeats."""
    seen = {}
    for match in names:
        code = match.species[0]
        seen.setdefault(code, (code, match.common_names[0] if match.common_names else code,
                               match.scientific_names[0] if match.scientific_names else ""))
    return list(seen.values())


def status_reply(names: list[Match], snapshot: dict | None) -> Reply:
    species = _named_species(names)
    if not species:
        return Reply(NEED_SPECIES[STATUS], declined=True, reason=SPECIES_NEEDED)
    if snapshot is None:
        return Reply(NO_SNAPSHOT, declined=True, reason=STATUS_UNAVAILABLE)
    lines, sources = [], []
    for code, name, scientific in species:
        entry = snapshot["species"].get(code) or {}
        category = entry.get("category")
        if not category:
            names_checked = ", ".join(dict.fromkeys(
                filter(None, [scientific, entry.get("scientific_name"), entry.get("iucn_name")])))
            lines.append(f"IUCN has no global Red List assessment for the {name}"
                         + (f" under the names I checked ({names_checked})." if names_checked
                            else "."))
            continue
        number = len(sources) + 1
        full = CATEGORIES.get(category, category)
        year = entry.get("year_published") or "unknown year"
        lines.append(f"The {name} is listed as {full} ({category}) on the IUCN Red List, in "
                     f"its global assessment published in {year} [{number}].")
        sources.append({"number": number, "species_code": code, "common_name": name,
                        "scientific_name": scientific or entry.get("scientific_name") or "",
                        "section": f"global assessment, {year}", "origin": "iucn",
                        "url": entry.get("assessment_url") or "https://www.iucnredlist.org/",
                        "score": 1.0, "text": f"{full} ({category}), published {year}."})
    fetched = (snapshot.get("fetched") or {}).get("to")
    if sources and fetched:
        lines.append(f"(Red List data retrieved {fetched}.)")
    return Reply("\n".join(lines), sources=sources)


def _sighting_line(obs: dict) -> str:
    place = "a private location" if obs.get("location_private") else obs["location"]
    count = obs.get("count")
    birds = f" ({count} bird{'s' if count != 1 else ''})" if isinstance(count, int) else ""
    return f"- {obs['observed_at']}, {place}{birds}"


def live_reply(names: list[Match], sightings=recent_sightings) -> Reply:
    species = _named_species(names)
    if not species:
        return Reply(NEED_SPECIES[LIVE], declined=True, reason=SPECIES_NEEDED)
    days, region = config.RECENT_DAYS, config.REGION_NAME
    blocks, sources = [], []
    for code, name, scientific in species:
        try:
            observations = sightings(code)
        except (httpx.HTTPError, RuntimeError) as error:
            # The user gets a plain apology; the log gets the cause (a 403 is a bad
            # key, a timeout a slow eBird). No secrets: the key travels in a header.
            status = getattr(getattr(error, "response", None), "status_code", None)
            log.warning("eBird lookup failed for %s: %s%s", code, type(error).__name__,
                        f" (HTTP {status})" if status else f" ({error})")
            return Reply(NO_EBIRD, declined=True, reason=LIVE_UNAVAILABLE)
        number = len(sources) + 1
        if not observations:
            # Empty is not "absent": coverage is sparse, and eBird withholds reports
            # of sensitive species (often exactly the notable ones) from this endpoint.
            text = (f"There are no public eBird reports of the {name} in {region} in the last "
                    f"{days} days [{number}]. eBird coverage of {region} is sparse, and eBird "
                    "withholds reports of some sensitive species, so this doesn't mean the bird "
                    "isn't there.")
            detail = "No public reports."
        else:
            shown = observations[:config.RECENT_SIGHTINGS_SHOWN]
            places = len(observations)
            text = (f"eBird has recent reports of the {name} from {places} "
                    f"location{'s' if places != 1 else ''} in {region} in the last {days} days "
                    f"[{number}]. Most recent first:\n" + "\n".join(map(_sighting_line, shown)))
            detail = "\n".join(map(_sighting_line, observations))
        blocks.append(text)
        sources.append({"number": number, "species_code": code, "common_name": name,
                        "scientific_name": scientific,
                        "section": f"recent sightings, last {days} days", "origin": "ebird_live",
                        "url": f"https://ebird.org/species/{code}/{config.REGION_CODE}",
                        "score": 1.0, "text": detail})
    return Reply("\n\n".join(blocks), sources=sources)
