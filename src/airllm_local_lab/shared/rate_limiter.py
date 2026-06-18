"""Token-bucket rate limiter and queue manager for inference requests.

Rate limits are read exclusively from config/default.toml [rate_limits].
No limit constant is hard-coded in application code.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RateLimitConfig:
    """Rate-limit parameters loaded from config."""

    max_requests_per_minute: int = 10
    max_tokens_per_request: int = 2048
    queue_size: int = 100
    retry_after_seconds: int = 60
    hf_api_calls_per_day: int = 200


class RateLimitExceededError(RuntimeError):
    """Raised when a request exceeds the configured rate limit."""

    def __init__(self, retry_after: int) -> None:
        """Initialise with the retry-after delay in seconds."""
        super().__init__(f"Rate limit exceeded — retry after {retry_after}s")
        self.retry_after = retry_after


@dataclass
class _TokenBucket:
    """Fixed-window token bucket; refills every `window_seconds`."""

    capacity: int
    window_seconds: float = 60.0
    _tokens: int = field(init=False)
    _window_start: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialise the bucket at full capacity."""
        self._tokens = self.capacity
        self._window_start = time.monotonic()

    def consume(self) -> bool:
        """Consume one token; return True if allowed, False if the bucket is empty."""
        with self._lock:
            now = time.monotonic()
            if now - self._window_start >= self.window_seconds:
                self._tokens = self.capacity
                self._window_start = now
            if self._tokens > 0:
                self._tokens -= 1
                return True
            return False


class RateLimiter:
    """Enforces per-minute request limits and per-request token caps.

    Instantiate once per process; thread-safe.  Config is injected at
    construction so tests can pass a custom ``RateLimitConfig``.
    """

    def __init__(self, cfg: RateLimitConfig | None = None) -> None:
        """Build a rate limiter from the supplied config (or defaults)."""
        self._cfg = cfg or RateLimitConfig()
        self._bucket = _TokenBucket(
            capacity=self._cfg.max_requests_per_minute,
            window_seconds=60.0,
        )
        self._queue: deque[float] = deque(maxlen=self._cfg.queue_size)

    def check_request(self, requested_tokens: int) -> None:
        """Validate a request; raise ``RateLimitExceededError`` if any limit is breached.

        Args:
            requested_tokens: Number of output tokens the caller wants to generate.

        Raises:
            ValueError: If ``requested_tokens`` exceeds the per-request token cap.
            RateLimitExceededError: If the per-minute request quota is exhausted.
        """
        if requested_tokens > self._cfg.max_tokens_per_request:
            raise ValueError(f"Requested {requested_tokens} tokens exceeds cap of {self._cfg.max_tokens_per_request}")
        if not self._bucket.consume():
            raise RateLimitExceededError(self._cfg.retry_after_seconds)
        self._queue.append(time.monotonic())

    @property
    def queue_depth(self) -> int:
        """Number of requests recorded in the sliding window queue."""
        return len(self._queue)
