import httpx

from chattfinch.clients import wikipedia_client
from conftest import load_fixture

URL = wikipedia_client.BASE_URL + wikipedia_client.API_PATH


def test_article_sections_follows_redirect_and_nests_subsections(respx_mock):
    respx_mock.get(URL).mock(
        return_value=httpx.Response(200, json=load_fixture("wikipedia/extract_nightjar.json")))
    article = wikipedia_client.article_sections("Caprimulgus europaeus")
    assert article["title"] == "European nightjar"
    # The fixture predates revision IDs, so the URL falls back to the title.
    assert article["revision_id"] is None
    assert article["url"] == "https://en.wikipedia.org/wiki/European_nightjar"
    sections = article["sections"]
    assert sections["Introduction"].startswith("The European nightjar")
    assert "Behaviour > Feeding" in sections
    assert "Description > Voice" in sections


def test_article_sections_revision_permalink(respx_mock):
    data = load_fixture("wikipedia/extract_nightjar.json")
    data["query"]["pages"][0]["revisions"] = [
        {"revid": 123, "parentid": 122, "timestamp": "2026-08-29T15:56:19Z"}]
    route = respx_mock.get(URL).mock(return_value=httpx.Response(200, json=data))
    article = wikipedia_client.article_sections("European nightjar")
    assert article["revision_id"] == 123
    assert article["revision_timestamp"] == "2026-08-29T15:56:19Z"
    assert article["url"] == "https://en.wikipedia.org/w/index.php?oldid=123"
    assert route.calls.last.request.url.params["prop"] == "extracts|revisions"


def test_article_sections_drops_reference_sections_and_their_children():
    extract = ("Lead text.\n\n== Description ==\nBrown.\n\n== Empty ==\n\n"
               "== References ==\nRef.\n\n=== Citations ===\nCite.\n\n"
               "== External links ==\n\n=== Poems ===\nPoem.")
    assert wikipedia_client._split_sections(extract) == {
        "Introduction": "Lead text.", "Description": "Brown."}


def test_article_sections_heading_levels_reset_path():
    extract = "== A ==\na\n=== B ===\nb\n==== C ====\nc\n=== D ===\nd\n== E ==\ne"
    assert list(wikipedia_client._split_sections(extract)) == [
        "A", "A > B", "A > B > C", "A > D", "E"]


def test_article_sections_missing_page_returns_none(respx_mock):
    respx_mock.get(URL).mock(
        return_value=httpx.Response(200, json=load_fixture("wikipedia/extract_missing.json")))
    assert wikipedia_client.article_sections("Notabird fakeus") is None


def test_article_redirects_follows_continuation(respx_mock):
    route = respx_mock.get(URL).mock(side_effect=[
        httpx.Response(200, json=load_fixture("wikipedia/redirects_page1.json")),
        httpx.Response(200, json=load_fixture("wikipedia/redirects_page2.json"))])
    result = wikipedia_client.article_redirects("European nightjar")
    assert result == {"title": "European nightjar", "section": None, "redirects": [
        "Eurasian Nightjar", "Caprimulgus europaeus", "Vaurie's nightjar",
        "Caprimulgus centralasicus"]}
    assert route.calls.last.request.url.params["rdcontinue"] == "12421648"


def test_article_redirects_flags_redirect_to_section(respx_mock):
    respx_mock.get(URL).mock(return_value=httpx.Response(200, json={"query": {
        "redirects": [{"from": "Genus species", "to": "Genus", "tofragment": "Species"}],
        "pages": [{"ns": 0, "title": "Genus", "redirects": [{"ns": 0, "title": "Other bird"}]}]}}))
    result = wikipedia_client.article_redirects("Genus species")
    assert result["title"] == "Genus" and result["section"] == "Species"


def test_article_redirects_missing_page_returns_none(respx_mock):
    respx_mock.get(URL).mock(
        return_value=httpx.Response(200, json=load_fixture("wikipedia/extract_missing.json")))
    assert wikipedia_client.article_redirects("Notabird fakeus") is None


def test_requests_send_user_agent(respx_mock):
    route = respx_mock.get(URL).mock(
        return_value=httpx.Response(200, json=load_fixture("wikipedia/extract_missing.json")))
    wikipedia_client._client = None  # build a real client so the configured headers apply
    wikipedia_client.article_sections("x")
    assert route.calls.last.request.headers["User-Agent"].startswith("Chattfinch")
