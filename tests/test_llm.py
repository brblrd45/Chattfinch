"""The model client against a fake Anthropic client: nothing is sent anywhere."""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from chattfinch import config, llm


def text(value):
    return SimpleNamespace(type="text", text=value)


def response(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


@pytest.fixture
def fake_client(monkeypatch):
    """Replaces the Anthropic client; set `.result` to a response or an exception."""
    fake = SimpleNamespace(calls=[], result=response(text("Hello.")))

    def create(**kwargs):
        fake.calls.append(kwargs)
        if isinstance(fake.result, Exception):
            raise fake.result
        return fake.result

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    return fake


def api_error(cls, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(status, request=request), body=None)


def test_request_shape(fake_client):
    assert llm.generate("SYSTEM", "USER") == "Hello."
    (call,) = fake_client.calls
    assert call["model"] == config.LLM_MODEL == "claude-haiku-5-5"
    assert call["system"] == "SYSTEM"
    assert call["messages"] == [{"role": "user", "content": "USER"}]
    assert call["output_config"] == {"effort": config.LLM_EFFORT}
    assert call["max_tokens"] == llm.MAX_TOKENS
    # Not sent: this model rejects non-default sampling parameters and thinking
    # budgets, and has no server-side refusal fallback.
    assert not {"temperature", "top_p", "top_k", "thinking", "fallbacks", "betas"} & set(call)


def test_text_blocks_are_joined_and_other_blocks_skipped(fake_client):
    fake_client.result = response(SimpleNamespace(type="thinking", thinking=""),
                                  text("Part one. "), text("Part two.\n"))
    assert llm.generate("s", "u") == "Part one. Part two."


@pytest.mark.parametrize("result, message", [
    (response(text("partial"), stop_reason="refusal"), "declined"),
    (response(text("cut off mid-sent"), stop_reason="max_tokens"), "cut off"),
    (response(), "no text"),
    (response(text("   ")), "no text"),
])
def test_unusable_responses_raise(fake_client, result, message):
    fake_client.result = result
    with pytest.raises(llm.LLMError, match=message):
        llm.generate("s", "u")


@pytest.mark.parametrize("error, message", [
    (api_error(anthropic.AuthenticationError, 401), "ANTHROPIC_API_KEY"),
    (api_error(anthropic.RateLimitError, 429), "rate-limited"),
    (api_error(anthropic.InternalServerError, 500), r"\(500\)"),
    (api_error(anthropic.BadRequestError, 400), r"\(400\)"),
    (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")),
     "network"),
])
def test_api_errors_become_llm_errors_with_a_clear_message(fake_client, error, message):
    fake_client.result = error
    with pytest.raises(llm.LLMError, match=message) as raised:
        llm.generate("s", "u")
    assert raised.value.__cause__ is error  # the original error is kept for debugging


@pytest.mark.parametrize("home", ["empty", "missing"])
def test_missing_credentials_are_reported_clearly(monkeypatch, tmp_path, home):
    """With no key anywhere, the real client fails before sending anything, and the
    message says how to fix it. The SDK raises different errors depending on whether
    its config directory exists (a bare TypeError if it does), so both are covered."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    directory = tmp_path if home == "empty" else tmp_path / "nonexistent"
    monkeypatch.setenv("HOME", str(directory))
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(directory / "anthropic"))
    llm._get_client.cache_clear()
    try:
        with pytest.raises(llm.LLMError, match="ANTHROPIC_API_KEY"):
            llm.generate("s", "u")
    finally:
        llm._get_client.cache_clear()


def test_unrelated_type_errors_are_not_swallowed(fake_client):
    fake_client.result = TypeError("create() got an unexpected keyword argument 'x'")
    with pytest.raises(TypeError, match="unexpected keyword"):
        llm.generate("s", "u")


def test_sdk_auth_type_error_becomes_a_clear_message(fake_client):
    fake_client.result = TypeError('"Could not resolve authentication method. Expected one of '
                                   'api_key, auth_token, or credentials to be set."')
    with pytest.raises(llm.LLMError, match="ANTHROPIC_API_KEY"):
        llm.generate("s", "u")
