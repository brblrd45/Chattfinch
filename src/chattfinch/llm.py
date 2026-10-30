"""The generation model: one function, `generate(system, user) -> str`.

This is the only module that knows which LLM provider is used. `rag.answer` takes
any function with this signature, so tests pass a fake and another provider can be
swapped in without touching retrieval or prompts.

Credentials come from the environment (ANTHROPIC_API_KEY in .env, or an
`ant auth login` profile); the client is created on first use.
"""

from functools import lru_cache

import anthropic

from chattfinch import config

MAX_TOKENS = 16000  # ceiling for thinking + answer (a cap, not a cost); answers are short

NO_CREDENTIALS = ("No valid Anthropic credentials. Set ANTHROPIC_API_KEY in .env "
                  "(or run `ant auth login`).")


class LLMError(RuntimeError):
    """The model could not produce an answer; the message says why."""


@lru_cache(maxsize=1)
def _get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def generate(system: str, user: str) -> str:
    """The model's text reply to one user message under `system`."""
    try:
        # No `fallbacks` parameter: Claude Haiku 5.5 has no server-side refusal
        # fallback, so a refusal is handled below instead.
        response = _get_client().messages.create(
            model=config.LLM_MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            output_config={"effort": config.LLM_EFFORT},
            messages=[{"role": "user", "content": user}],
        )
    except (anthropic.AuthenticationError, anthropic.CredentialsError) as error:
        raise LLMError(NO_CREDENTIALS) from error
    except TypeError as error:
        # With no key and no login profile at all, the SDK raises a bare TypeError
        # before sending anything.
        if "authentication method" not in str(error):
            raise
        raise LLMError(NO_CREDENTIALS) from error
    except anthropic.RateLimitError as error:
        raise LLMError("The model is rate-limited right now; try again shortly.") from error
    except anthropic.APIStatusError as error:
        raise LLMError(f"The model API returned an error ({error.status_code}): "
                       f"{error.message}") from error
    except anthropic.APIConnectionError as error:
        raise LLMError("Could not reach the model API; check the network connection.") from error

    if response.stop_reason == "refusal":
        raise LLMError("The model declined to answer this request.")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if response.stop_reason == "max_tokens":
        raise LLMError("The model's reply was cut off at the token limit.")
    if not text:
        raise LLMError("The model returned no text.")
    return text
