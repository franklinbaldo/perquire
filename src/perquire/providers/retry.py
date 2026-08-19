"""Small bounded retry primitive for transient upstream provider failures."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
_TRANSIENT_NAMES = {"RateLimitError", "ServiceUnavailableError", "EmptyProviderResponse"}
_RETRY_AFTER = re.compile(r"retry_after_seconds(?:_raw)?[\"']?\s*[:=]\s*[\"']?([0-9]+(?:\.[0-9]+)?)")


class EmptyProviderResponse(RuntimeError):
    """The upstream accepted the request but produced no usable text."""


def is_transient_provider_error(error: Exception) -> bool:
    """Return true only for retryable 429/503/empty-output provider failures."""
    if type(error).__name__ in _TRANSIENT_NAMES:
        return True
    text = str(error).lower()
    return (
        "429" in text
        or "503" in text
        or "rate limit" in text
        or "temporarily unavailable" in text
    )


def retry_after_seconds(error: Exception, *, default: float = 5.0, maximum: float = 30.0) -> float:
    """Extract OpenRouter/LiteLLM retry guidance without depending on one exception shape."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or getattr(error, "headers", None)
    if headers:
        value = headers.get("Retry-After") or headers.get("retry-after")
        try:
            if value is not None:
                return min(maximum, max(0.0, float(value)))
        except (TypeError, ValueError):
            pass

    match = _RETRY_AFTER.search(str(error))
    if match:
        return min(maximum, max(0.0, float(match.group(1))))
    return min(maximum, max(0.0, default))


def call_with_transient_retries(
    operation: Callable[[], T],
    *,
    before_attempt: Callable[[], None] | None = None,
    max_retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run an operation with bounded retries for transient provider failures.

    ``before_attempt`` is invoked for every network attempt, including retries,
    so the shared request pacer continues to account for retry traffic.
    """
    for attempt in range(max_retries + 1):
        if before_attempt is not None:
            before_attempt()
        try:
            return operation()
        except Exception as error:
            if attempt >= max_retries or not is_transient_provider_error(error):
                raise
            sleep(retry_after_seconds(error))
    raise AssertionError("unreachable")
