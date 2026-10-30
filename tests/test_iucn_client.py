import httpx
import pytest

from chattfinch import config
from chattfinch.clients import iucn_client
from conftest import load_fixture

URL = f"{iucn_client.BASE_URL}/taxa/scientific_name"


def test_species_status_uses_latest_global_assessment(respx_mock):
    # The fixture's newest assessment (2023, EN) is Europe-only; the latest global
    # one is 2021, NT.
    route = respx_mock.get(URL).mock(
        return_value=httpx.Response(200, json=load_fixture("iucn/synthetic_species.json")))
    status = iucn_client.species_status("Fictus avis")
    assert status["category"] == "NT"
    assert status["year_published"] == "2021"
    assert status["assessment_url"].startswith("https://www.iucnredlist.org/species/")
    params = route.calls.last.request.url.params
    assert (params["genus_name"], params["species_name"]) == ("Fictus", "avis")
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-token"


def test_species_status_english_names_main_first(respx_mock):
    respx_mock.get(URL).mock(
        return_value=httpx.Response(200, json=load_fixture("iucn/synthetic_species.json")))
    assert iucn_client.species_status("Fictus avis")["english_names"] == [
        "Fictional Lapwing", "Pretend Plover"]


def test_species_status_without_global_assessment(respx_mock):
    data = load_fixture("iucn/synthetic_species.json")
    data["assessments"] = [a for a in data["assessments"]
                           if all(s["code"] != "1" for s in a["scopes"])]
    respx_mock.get(URL).mock(return_value=httpx.Response(200, json=data))
    status = iucn_client.species_status("Fictus avis")
    assert status["category"] is None and status["year_published"] is None


def test_species_status_unknown_name_returns_none(respx_mock):
    # IUCN files Demoiselle Crane as Anthropoides virgo, so eBird's Grus virgo is a 404.
    respx_mock.get(URL).mock(
        return_value=httpx.Response(404, json=load_fixture("iucn/not_found.json")))
    assert iucn_client.species_status("Grus virgo") is None


def test_species_status_other_errors_raise(respx_mock):
    respx_mock.get(URL).mock(return_value=httpx.Response(401))
    with pytest.raises(httpx.HTTPStatusError):
        iucn_client.species_status("Fictus avis")


def test_missing_token_gives_clear_error(monkeypatch):
    monkeypatch.setattr(iucn_client, "_client", None)
    monkeypatch.setattr(config, "IUCN_API_TOKEN", None)
    with pytest.raises(RuntimeError, match="IUCN_API_TOKEN"):
        iucn_client.species_status("Fictus avis")
