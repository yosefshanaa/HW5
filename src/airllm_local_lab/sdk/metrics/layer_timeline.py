"""Per-layer load/compute timeline recorder — makes the I/O bottleneck visible."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class LayerEvent:
    layer_idx: int
    load_start: float
    load_end: float
    compute_end: float

    @property
    def load_ms(self) -> float:
        return (self.load_end - self.load_start) * 1000

    @property
    def compute_ms(self) -> float:
        return (self.compute_end - self.load_end) * 1000

    @property
    def total_ms(self) -> float:
        return (self.compute_end - self.load_start) * 1000


@dataclass
class LayerTimeline:
    events: list[LayerEvent] = field(default_factory=list)

    def record(self, layer_idx: int, load_start: float, load_end: float, compute_end: float) -> None:
        self.events.append(LayerEvent(layer_idx, load_start, load_end, compute_end))

    def io_fraction(self) -> float:
        if not self.events:
            return 0.0
        total_load = sum(e.load_ms for e in self.events)
        total_all = sum(e.total_ms for e in self.events)
        return total_load / total_all if total_all else 0.0

    def to_dicts(self) -> list[dict]:
        return [
            {
                "layer": e.layer_idx,
                "load_ms": round(e.load_ms, 3),
                "compute_ms": round(e.compute_ms, 3),
                "total_ms": round(e.total_ms, 3),
            }
            for e in self.events
        ]


class LayerTimelineRecorder:
    """Used as a context manager around per-layer load/compute."""

    def __init__(self, timeline: LayerTimeline, layer_idx: int) -> None:
        self._tl = timeline
        self._idx = layer_idx
        self._load_start = 0.0
        self._load_end = 0.0

    def loaded(self) -> None:
        self._load_end = time.perf_counter()

    def __enter__(self) -> LayerTimelineRecorder:
        self._load_start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        compute_end = time.perf_counter()
        if self._load_end == 0.0:
            self._load_end = self._load_start
        self._tl.record(self._idx, self._load_start, self._load_end, compute_end)
