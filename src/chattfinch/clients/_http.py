"""Shared HTTP client with retries and simple rate limiting, used by every API client."""

import time
from collections.abc import Callable

import httpx

RETRY_STATUSES = {429, 500, 502, 503, 504}


class ApiClient:
    """GETs JSON from one API. Retries transient failures (429/5xx and connection
    errors) with exponential backoff, honoring a numeric Retry-After header, and
    waits at least `min_interval_s` between requests."""

    def __init__(self, base_url: str, headers: dict[str, str] | None = None,
                 min_interval_s: float = 0.0, max_retries: int = 3,
                 backoff_s: float = 1.0, timeout_s: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep):
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout_s)
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._sleep = sleep
        self._last_request_at: float | None = None

    def get_json(self, path: str, params: dict | None = None):
        for attempt in range(self.max_retries + 1):
            self._wait_for_slot()
            try:
                response = self._client.get(path, params=params)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
                self._sleep(self.backoff_s * 2 ** attempt)
                continue
            if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                self._sleep(self._retry_delay(response, attempt))
                continue
            response.raise_for_status()
            return response.json()

    def _wait_for_slot(self) -> None:
        now = time.monotonic()
        if self._last_request_at is not None:
            remaining = self.min_interval_s - (now - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = time.monotonic()

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.isdigit():
            return float(retry_after)
        return self.backoff_s * 2 ** attempt
