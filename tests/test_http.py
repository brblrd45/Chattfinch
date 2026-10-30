import httpx
import pytest

from chattfinch.clients._http import ApiClient

URL = "https://api.example.org/thing"


def make_client(sleeps: list[float], **kwargs) -> ApiClient:
    return ApiClient("https://api.example.org", sleep=sleeps.append, **kwargs)


def test_retries_server_errors_then_succeeds(respx_mock):
    route = respx_mock.get(URL).mock(side_effect=[
        httpx.Response(503), httpx.Response(502), httpx.Response(200, json={"ok": True})])
    sleeps = []
    assert make_client(sleeps, backoff_s=1.0).get_json("/thing") == {"ok": True}
    assert route.call_count == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff


def test_honors_numeric_retry_after(respx_mock):
    respx_mock.get(URL).mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json={})])
    sleeps = []
    make_client(sleeps).get_json("/thing")
    assert sleeps == [7.0]


def test_gives_up_after_max_retries(respx_mock):
    route = respx_mock.get(URL).mock(return_value=httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        make_client([], max_retries=2).get_json("/thing")
    assert route.call_count == 3


def test_does_not_retry_client_errors(respx_mock):
    route = respx_mock.get(URL).mock(return_value=httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        make_client([]).get_json("/thing")
    assert route.call_count == 1


def test_retries_connection_errors(respx_mock):
    respx_mock.get(URL).mock(side_effect=[httpx.ConnectError("down"),
                                          httpx.Response(200, json={"ok": True})])
    assert make_client([]).get_json("/thing") == {"ok": True}


def test_rate_limit_waits_between_requests(respx_mock):
    respx_mock.get(URL).mock(return_value=httpx.Response(200, json={}))
    sleeps = []
    client = make_client(sleeps, min_interval_s=10.0)
    client.get_json("/thing")
    client.get_json("/thing")
    assert len(sleeps) == 1 and 9.0 < sleeps[0] <= 10.0
