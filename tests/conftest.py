import json
from pathlib import Path

import pytest

from chattfinch.clients import ebird_client, gbif_client, iucn_client, wikipedia_client
from chattfinch.clients._http import ApiClient

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(relative_path: str) -> dict:
    """A recorded (trimmed) API response from tests/fixtures/."""
    return json.loads((FIXTURES / relative_path).read_text())


@pytest.fixture(autouse=True)
def no_wait_clients(monkeypatch):
    """Fresh module clients per test that never actually sleep. Keyed clients get a
    dummy key, so tests never use (or need) the real ones in .env."""
    no_sleep = lambda seconds: None
    monkeypatch.setattr(ebird_client, "_client", ApiClient(
        ebird_client.BASE_URL, headers={"X-eBirdApiToken": "test-key"}, sleep=no_sleep))
    monkeypatch.setattr(ebird_client, "_live_client", ApiClient(
        ebird_client.BASE_URL, headers={"X-eBirdApiToken": "test-key"}, max_retries=1,
        sleep=no_sleep))
    monkeypatch.setattr(iucn_client, "_client", ApiClient(
        iucn_client.BASE_URL, headers={"Authorization": "Bearer test-token"}, sleep=no_sleep))
    monkeypatch.setattr(gbif_client, "_client", ApiClient(gbif_client.BASE_URL, sleep=no_sleep))
    monkeypatch.setattr(wikipedia_client, "_client",
                        ApiClient(wikipedia_client.BASE_URL, sleep=no_sleep))


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False,
                     help="also run tests marked 'live', which call the paid LLM API")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="calls the paid LLM API; run with --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
