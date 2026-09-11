from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Optional

import httpx

from .rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)


# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# HTTP status codes that indicate permanent errors (no retry)
PERMANENT_ERROR_CODES = {400, 401, 403, 404, 405, 422}


class RateLimitedAsyncHTTP:
    """Async HTTP client with rate limiting, concurrency control, and retry logic.

    Uses httpx.AsyncClient for true async HTTP requests.
    Respects Retry-After headers and uses exponential backoff with jitter.
    """

    def __init__(
        self,
        requests_per_minute: int = 25,
        max_retries: int = 3,
        max_concurrency: int = 5,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.max_concurrency = max_concurrency
        self.timeout = timeout or httpx.Timeout(connect=10, read=45, write=10, pool=10)
        self._rate_limiter = TokenBucketRateLimiter(requests_per_minute)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a POST request with rate limiting and retry logic.

        Acquires the semaphore before making the request and respects
        rate limits. Retries on 429 and 5xx errors with backoff.
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            # Rate limit before making request
            await self._rate_limiter.acquire_async()

            # Concurrency control
            async with self._semaphore:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, json=json, **kwargs)

                    # Success
                    if response.status_code < 400:
                        return response

                    # Permanent error - don't retry
                    if response.status_code in PERMANENT_ERROR_CODES:
                        response.raise_for_status()

                    # Retryable error
                    if response.status_code in RETRYABLE_STATUS_CODES:
                        last_exception = httpx.HTTPStatusError(
                            f"HTTP {response.status_code}",
                            request=response.request,
                            response=response,
                        )

                        if attempt < self.max_retries:
                            # Check for Retry-After header
                            retry_after = self._parse_retry_after(response)
                            if retry_after is not None:
                                wait_time = retry_after
                            else:
                                # Exponential backoff with jitter: 1s, 2s, 4s + jitter
                                wait_time = (2 ** attempt) + random.uniform(0, 0.5)

                            logger.warning(
                                "Rate limit or server error (HTTP %d). Retrying in %.1fs... (attempt %d/%d)",
                                response.status_code,
                                wait_time,
                                attempt + 1,
                                self.max_retries,
                            )
                            await asyncio.sleep(wait_time)
                            continue

                    # Unknown error - raise immediately
                    response.raise_for_status()

        # Exhausted retries
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Request failed after {self.max_retries + 1} attempts")

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float | None:
        """Parse Retry-After header value. Returns seconds or None."""
        retry_after = response.headers.get("retry-after")
        if retry_after is None:
            return None
        try:
            # Try as seconds first
            return float(retry_after)
        except ValueError:
            pass
        try:
            # Try as HTTP date
            from email.utils import parsedate_to_datetime
            retry_date = parsedate_to_datetime(retry_after)
            now = time.time()
            return max(0.0, retry_date.timestamp() - now)
        except (ValueError, TypeError):
            return None

    async def close(self) -> None:
        """No-op for compatibility; clients are created per-request."""
        pass
