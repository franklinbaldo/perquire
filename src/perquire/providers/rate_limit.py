"""Client-side pacing for provider APIs with published request rates."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

_SECONDS_PER_MINUTE = 60.0


@dataclass
class RequestPacer:
    """Space outgoing requests so a published per-minute rate is not exceeded.

    OpenRouter's free tier allows 20 requests per minute, and a benchmark sweep
    issues far more than that back to back. Pacing locally keeps the run inside
    the published rate instead of relying on retries after rejection.

    A rate of zero disables pacing, which keeps tests free of real waiting.
    """

    requests_per_minute: int
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _last_request_at: float | None = field(default=None, init=False, repr=False)

    @property
    def min_interval(self) -> float:
        if self.requests_per_minute <= 0:
            return 0.0
        return _SECONDS_PER_MINUTE / self.requests_per_minute

    def wait(self) -> None:
        """Block until the next request may be sent."""
        interval = self.min_interval
        if interval <= 0.0:
            return

        now = self.clock()
        if self._last_request_at is not None:
            remaining = interval - (now - self._last_request_at)
            if remaining > 0.0:
                self.sleep(remaining)
                now += remaining
        self._last_request_at = now
