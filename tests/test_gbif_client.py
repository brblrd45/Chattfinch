import httpx

from chattfinch.clients import gbif_client
from conftest import load_fixture

MATCH_URL = f"{gbif_client.BASE_URL}/species/match"


def test_match_name_exact(respx_mock):
    route = respx_mock.get(MATCH_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("gbif/match_exact.json")))
    match = gbif_client.match_name("Caprimulgus europaeus")
    assert match == {
        "usage_key": 8109681, "accepted_key": 8109681,
        "canonical_name": "Caprimulgus europaeus", "accepted_name": "Caprimulgus europaeus",
        "status": "ACCEPTED", "match_type": "EXACT", "confidence": 100,
    }
    assert route.calls.last.request.url.params["class"] == "Aves"


def test_match_name_synonym_points_to_accepted_taxon(respx_mock):
    respx_mock.get(MATCH_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("gbif/match_synonym.json")))
    match = gbif_client.match_name("Grus virgo")
    assert match["status"] == "SYNONYM"
    assert match["usage_key"] != match["accepted_key"]
    assert match["accepted_name"] == "Anthropoides virgo"


def test_match_name_unknown_returns_none(respx_mock):
    # GBIF answers an unknown name with a HIGHERRANK match to Aves, not an empty result.
    respx_mock.get(MATCH_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("gbif/match_higherrank.json")))
    assert gbif_client.match_name("Notabird fakeus") is None


def test_vernacular_names_filters_language_and_dedupes(respx_mock):
    respx_mock.get(f"{gbif_client.BASE_URL}/species/8109681/vernacularNames").mock(
        return_value=httpx.Response(200, json=load_fixture("gbif/vernacular_names.json")))
    names = gbif_client.vernacular_names(8109681)
    assert "Goatsucker" in names and "European Nightjar" in names
    assert len(names) == len(set(names))
    assert all(name.isascii() for name in names)  # Russian/German records filtered out


def test_vernacular_names_follows_pagination(respx_mock):
    page = lambda names, end: httpx.Response(200, json={
        "endOfRecords": end,
        "results": [{"vernacularName": n, "language": "eng"} for n in names]})
    route = respx_mock.get(f"{gbif_client.BASE_URL}/species/1/vernacularNames").mock(
        side_effect=[page(["A", "B"], False), page(["B", "C"], True)])
    assert gbif_client.vernacular_names(1) == ["A", "B", "C"]
    assert route.calls.last.request.url.params["offset"] == str(gbif_client.PAGE_SIZE)
