"""English Wikipedia client (MediaWiki Action API): article sections and redirect titles."""

import re

from chattfinch import config
from chattfinch.clients._http import ApiClient

BASE_URL = "https://en.wikipedia.org/w"
API_PATH = "/api.php"
LEAD_SECTION = "Introduction"
# Sections with no descriptive content about the bird.
SKIPPED_SECTIONS = {"references", "citations", "cited texts", "notes", "explanatory notes",
                    "external links", "see also", "further reading", "sources", "bibliography",
                    "gallery"}
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)

_client: ApiClient | None = None


def _get_client() -> ApiClient:
    global _client
    if _client is None:
        _client = ApiClient(BASE_URL, headers={"User-Agent": config.WIKIPEDIA_USER_AGENT},
                            min_interval_s=0.5)
    return _client


def _query(params: dict) -> dict:
    return _get_client().get_json(API_PATH, params={"action": "query", "format": "json",
                                              "formatversion": 2, **params})


def article_sections(title: str) -> dict | None:
    """Plain-text sections of an article, following redirects (so a scientific name
    works as `title`). Returns None if there is no such article, else
    {"title", "revision_id", "revision_timestamp", "url", "sections": {path: text}}.
    `url` is a permanent link to the exact revision, for CC BY-SA attribution.

    Subsection paths join their headings ("Behaviour > Feeding"). Reference and
    link sections, and subsections under them, are dropped, as are empty sections.
    """
    data = _query({"prop": "extracts|revisions", "rvprop": "ids|timestamp",
                   "explaintext": 1, "exsectionformat": "wiki", "redirects": 1,
                   "titles": title})
    page = data["query"]["pages"][0]
    if page.get("missing") or page.get("invalid"):
        return None
    revision = (page.get("revisions") or [{}])[0]
    revision_id = revision.get("revid")
    return {
        "title": page["title"],
        "revision_id": revision_id,
        "revision_timestamp": revision.get("timestamp"),
        "url": f"https://en.wikipedia.org/w/index.php?oldid={revision_id}" if revision_id else
               f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}",
        "sections": _split_sections(page.get("extract", "")),
    }


def _split_sections(extract: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    path: list[tuple[int, str]] = []  # (level, heading) of the current section's ancestors
    current, start = LEAD_SECTION, 0

    def add(name: str, text: str) -> None:
        text = text.strip()
        if text and not any(h.lower() in SKIPPED_SECTIONS for h in name.split(" > ")):
            sections[name] = f"{sections[name]}\n\n{text}" if name in sections else text

    for match in _HEADING.finditer(extract):
        add(current, extract[start:match.start()])
        level, heading = len(match.group(1)), match.group(2)
        path = [(lvl, h) for lvl, h in path if lvl < level] + [(level, heading)]
        current, start = " > ".join(h for _, h in path), match.end()
    add(current, extract[start:])
    return sections


def article_redirects(title: str) -> dict | None:
    """The article `title` resolves to, and the main-namespace titles that redirect
    to it. Returns {"title", "section", "redirects"} or None if there is no article.

    `section` is set when `title` redirects to a section of another article (e.g. a
    species covered inside its genus article). The redirects then belong to that
    whole article, not to this species, so callers shouldn't use them as aliases.
    Redirect titles are raw alias candidates: they include subspecies, former names
    and oddities.
    """
    params = {"prop": "redirects", "rdnamespace": 0, "rdlimit": "max", "redirects": 1,
              "titles": title}
    result = None
    while True:
        data = _query(params)
        page = data["query"]["pages"][0]
        if page.get("missing") or page.get("invalid"):
            return None
        if result is None:
            fragments = [r.get("tofragment") for r in data["query"].get("redirects", [])]
            result = {"title": page["title"], "section": next(filter(None, fragments), None),
                      "redirects": []}
        result["redirects"].extend(r["title"] for r in page.get("redirects", []))
        if "continue" not in data:
            return result
        params = {**params, **data["continue"]}
