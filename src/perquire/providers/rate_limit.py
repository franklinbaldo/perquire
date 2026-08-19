"""Client-side pacing for provider APIs with published request rates."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

_SECONDS_PER_MINUTE = 60.0


@dataclass
class RequestPacer:
    """Space outgoing requests so a published per-minute rate is not exceeded.

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
        """Block until the next request may be sent.

        Re-read the clock after sleeping instead of assuming that ``sleep``
        advanced time by exactly the requested duration. This keeps injected
        clocks and early-returning sleeps from compounding waits artificially.
        """
        interval = self.min_interval
        if interval <= 0.0:
            return

        now = self.clock()
        if self._last_request_at is not None:
            elapsed = max(0.0, now - self._last_request_at)
            remaining = interval - elapsed
            if remaining > 0.0:
                self.sleep(remaining)
                now = self.clock()
        self._last_request_at = now


_SHARED_PACERS: dict[tuple[str, int], RequestPacer] = {}


def get_shared_pacer(namespace: str, requests_per_minute: int) -> RequestPacer:
    """Return one process-wide pacer for all calls sharing an API rate limit.

    Provider roles that use the same upstream quota (for example OpenRouter chat
    and embedding requests) must share the same pacer. Otherwise two independent
    20 RPM pacers can accidentally emit roughly 40 RPM in aggregate.

    The namespace identifies the upstream quota rather than the secret value, so
    credentials are never retained as registry keys.
    """
    key = (namespace, requests_per_minute)
    pacer = _SHARED_PACERS.get(key)
    if pacer is None:
        pacer = RequestPacer(requests_per_minute=requests_per_minute)
        _SHARED_PACERS[key] = pacer
    return pacer
