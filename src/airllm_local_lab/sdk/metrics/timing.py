"""TTFT, TPOT/ITL, throughput timing utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TimingResult:
    """Aggregate timing measurements for one inference call."""

    ttft_s: float = 0.0
    token_timestamps: list[float] = field(default_factory=list)
    total_s: float = 0.0

    @property
    def num_tokens(self) -> int:
        """Number of tokens recorded via ``record_token``."""
        return len(self.token_timestamps)

    @property
    def tpot_s(self) -> float:
        """Mean inter-token latency (TPOT) in seconds; 0.0 when fewer than 2 tokens."""
        if len(self.token_timestamps) < 2:
            return 0.0
        gaps = [self.token_timestamps[i] - self.token_timestamps[i - 1] for i in range(1, len(self.token_timestamps))]
        return sum(gaps) / len(gaps)

    @property
    def throughput_tps(self) -> float:
        """Tokens per second over the full generation wall-clock time."""
        if self.total_s <= 0:
            return 0.0
        return self.num_tokens / self.total_s


class Timer:
    """Context manager that records TTFT and per-token timestamps."""

    def __init__(self) -> None:
        """Initialise the timer in a stopped state."""
        self._start: float = 0.0
        self._first: float | None = None
        self._tokens: list[float] = []

    def start(self) -> None:
        """Begin a new timing session, resetting all prior state."""
        self._start = time.perf_counter()
        self._first = None
        self._tokens = []

    def record_token(self) -> None:
        """Stamp the arrival time of the next output token."""
        now = time.perf_counter()
        if self._first is None:
            self._first = now
        self._tokens.append(now - self._start)

    def finish(self) -> TimingResult:
        """Stop the timer and return a populated ``TimingResult``."""
        total = time.perf_counter() - self._start
        ttft = (self._first - self._start) if self._first is not None else total
        return TimingResult(ttft_s=ttft, token_timestamps=list(self._tokens), total_s=total)


def summarise(results: list[TimingResult]) -> dict[str, float]:
    """Compute median ± stdev statistics across a list of ``TimingResult`` objects."""
    import statistics

    if not results:
        return {}
    ttfts = [r.ttft_s for r in results]
    tpots = [r.tpot_s for r in results if r.tpot_s > 0]
    tpss = [r.throughput_tps for r in results if r.throughput_tps > 0]
    return {
        "ttft_median_s": statistics.median(ttfts),
        "ttft_stdev_s": statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
        "tpot_median_s": statistics.median(tpots) if tpots else 0.0,
        "throughput_median_tps": statistics.median(tpss) if tpss else 0.0,
    }
