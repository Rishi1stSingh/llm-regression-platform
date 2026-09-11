from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Token bucket rate limiter for controlling request frequency.

    Allows bursting up to the bucket size, then limits to the configured
    rate. Works with asyncio for use in async contexts.
    """

    def __init__(self, requests_per_minute: int = 25, burst_size: Optional[int] = None) -> None:
        self._rate = requests_per_minute / 60.0  # tokens per second
        self._burst = burst_size or max(1, requests_per_minute // 5)
        self._tokens = float(self._burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire_async(self) -> None:
        """Wait until a token is available (async version)."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._last_refill = now

                # Refill tokens based on elapsed time
                self._tokens = min(self._burst, self._tokens + elapsed * self._rate)

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                else:
                    # Wait until we have enough tokens
                    wait_time = (1.0 - self._tokens) / self._rate

            await asyncio.sleep(wait_time)

    def acquire(self) -> None:
        """Block until a token is available (sync version)."""
        import threading
        while True:
            with threading.RLock():
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._last_refill = now

                self._tokens = min(self._burst, self._tokens + elapsed * self._rate)

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                else:
                    wait_time = (1.0 - self._tokens) / self._rate

            time.sleep(wait_time)


class RateLimitedHTTPClient:
    """Synchronous HTTP client wrapper with rate limiting and retry logic.

    Wraps requests.Session to add:
    - Token bucket rate limiting (requests per minute)
    - Automatic retry on 429 and 5xx errors
    - Respect for Retry-After headers
    - Exponential backoff with jitter
    """

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    PERMANENT_ERROR_CODES = {400, 401, 403, 404, 405, 422}

    def __init__(
        self,
        requests_per_minute: int = 25,
        max_retries: int = 3,
        timeout: tuple[float, float] | float = (10, 45),
    ) -> None:
        import requests

        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.timeout = timeout
        self._rate_limiter = TokenBucketRateLimiter(requests_per_minute)
        self._session = requests.Session()

    def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        json: dict | None = None,
        **kwargs,
    ):
        """Make a POST request with rate limiting and retry logic."""
        import time
        import random

        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            # Rate limit before making request
            self._rate_limiter.acquire()

            response = self._session.post(
                url, headers=headers, json=json, timeout=self.timeout, **kwargs
            )

            # Success
            if response.status_code < 400:
                return response

            # Permanent error - don't retry
            if response.status_code in self.PERMANENT_ERROR_CODES:
                response.raise_for_status()

            # Retryable error
            if response.status_code in self.RETRYABLE_STATUS_CODES:
                last_exception = Exception(f"HTTP {response.status_code}")

                if attempt < self.max_retries:
                    retry_after = self._parse_retry_after(response)
                    if retry_after is not None:
                        wait_time = retry_after
                    else:
                        wait_time = (2 ** attempt) + random.uniform(0, 0.5)

                    logger.warning(
                        "Rate limit or server error (HTTP %d). Retrying in %.1fs... (attempt %d/%d)",
                        response.status_code,
                        wait_time,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(wait_time)
                    continue

            # Unknown error - raise immediately
            response.raise_for_status()

        # Exhausted retries
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Request failed after {self.max_retries + 1} attempts")

    @staticmethod
    def _parse_retry_after(response) -> float | None:
        """Parse Retry-After header value. Returns seconds or None."""
        retry_after = response.headers.get("retry-after")
        if retry_after is None:
            return None
        try:
            return float(retry_after)
        except ValueError:
            pass
        try:
            from email.utils import parsedate_to_datetime

            retry_date = parsedate_to_datetime(retry_after)
            import time as _time

            return max(0.0, retry_date.timestamp() - _time.time())
        except (ValueError, TypeError):
            return None

    def close(self) -> None:
        self._session.close()
