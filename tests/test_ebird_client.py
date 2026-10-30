import httpx
import pytest

from chattfinch import config
from chattfinch.clients import ebird_client
from conftest import load_fixture

BASE = ebird_client.BASE_URL


def test_region_species_codes(respx_mock):
    route = respx_mock.get(f"{BASE}/product/spplist/KZ").mock(
        return_value=httpx.Response(200, json=load_fixture("ebird/spplist_kz.json")))
    codes = ebird_client.region_species_codes("KZ")
    assert "soclap1" in codes and len(codes) == 6
    assert route.calls.last.request.headers["X-eBirdApiToken"] == "test-key"


def test_taxonomy_cleans_and_sorts(respx_mock):
    respx_mock.get(f"{BASE}/ref/taxonomy/ebird").mock(
        return_value=httpx.Response(200, json=load_fixture("ebird/taxonomy.json")))
    frame = ebird_client.taxonomy(load_fixture("ebird/spplist_kz.json"))
    assert list(frame.columns) == ["species_code", "common_name", "scientific_name", "category",
                                   "order", "family", "family_common_name", "taxon_order"]
    assert frame["taxon_order"].is_monotonic_increasing
    crane = frame.set_index("species_code").loc["demcra1"]
    assert crane["scientific_name"] == "Grus virgo" and crane["family"] == "Gruidae"
    assert set(frame["category"]) == {"species", "hybrid"}


def test_taxonomy_batches_long_code_lists(respx_mock, monkeypatch):
    monkeypatch.setattr(ebird_client, "TAXONOMY_BATCH", 4)
    route = respx_mock.get(f"{BASE}/ref/taxonomy/ebird").mock(
        return_value=httpx.Response(200, json=[]))
    ebird_client.taxonomy([f"code{i}" for i in range(10)])
    assert route.call_count == 3
    assert route.calls[2].request.url.params["species"] == "code8,code9"


def test_recent_observations(respx_mock):
    route = respx_mock.get(f"{BASE}/data/obs/KZ/recent").mock(
        return_value=httpx.Response(200, json=load_fixture("ebird/recent_obs_synthetic.json")))
    observations = ebird_client.recent_observations("KZ", back_days=7)
    assert route.calls.last.request.url.params["back"] == "7"
    assert observations[0]["common_name"] == "Sociable Lapwing" and observations[0]["count"] == 12
    assert observations[1]["count"] is None  # eBird omits howMany for "X" (present, uncounted)
    assert observations[1]["location_private"] is True


def test_recent_observations_for_one_species(respx_mock):
    route = respx_mock.get(f"{BASE}/data/obs/KZ/recent/soclap1").mock(
        return_value=httpx.Response(200, json=[]))
    assert ebird_client.recent_observations("KZ", "soclap1") == []
    assert route.called


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.setattr(ebird_client, "_client", None)
    monkeypatch.setattr(config, "EBIRD_API_KEY", None)
    with pytest.raises(RuntimeError, match="EBIRD_API_KEY"):
        ebird_client.region_species_codes("KZ")


def test_the_live_client_gives_up_quickly(monkeypatch):
    """The bot's query path: a short timeout and a single retry, not the batch
    client's patient backoff."""
    monkeypatch.setattr(ebird_client, "_live_client", None)
    monkeypatch.setattr(config, "EBIRD_API_KEY", "test-key")
    client = ebird_client._get_live_client()
    assert client.max_retries == 1
    assert client._client.timeout.read == config.LIVE_TIMEOUT_S


def test_recent_observations_retries_once_then_raises(respx_mock):
    route = respx_mock.get(f"{BASE}/data/obs/KZ/recent/soclap1").mock(
        return_value=httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        ebird_client.recent_observations("KZ", "soclap1")
    assert route.call_count == 2
