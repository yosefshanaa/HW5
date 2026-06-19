"""Capture REAL per-layer load/compute timing from AirLLM's MLX backend.

Monkey-patches the two hot spots in AirLLM's layer-streaming generate loop:
  * ``MlxModelPersister.load_model`` — the disk stream of one layer shard (load_ms)
  * ``TransformerBlock.__call__``    — that layer's forward compute (compute_ms)

MLX is lazily evaluated, so each timed region is forced with ``mx.eval`` to make
the disk read / matmuls actually happen inside the measured window. Only the
first forward pass over the layers (prompt processing) is recorded — one clean
sweep of every transformer layer.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from airllm_local_lab.sdk.metrics.layer_timeline import LayerTimeline

_LAYER_RE = re.compile(r"layers\.(\d+)$")


class _Capture:
    """Correlates each layer's disk load with its immediately following forward."""

    def __init__(self, timeline: LayerTimeline, eval_fn: Callable[[object], None]) -> None:
        self._tl = timeline
        self._eval = eval_fn
        self._idx: int | None = None
        self._load_start = 0.0
        self._load_end = 0.0
        self._seen: set[int] = set()

    def on_load(self, layer_name: str, do_load: Callable[[], object]) -> object:
        match = _LAYER_RE.search(layer_name)
        if match is None:
            return do_load()  # embed / norm / lm_head — not a transformer layer
        start = time.perf_counter()
        weights = do_load()
        self._eval(weights)  # force the mmap'd shard to actually stream from disk
        self._idx = int(match.group(1))
        self._load_start, self._load_end = start, time.perf_counter()
        return weights

    def on_forward(self, output: object) -> object:
        # output is the (hidden, cache) tuple from the just-run TransformerBlock
        if self._idx is not None and self._idx not in self._seen:
            self._eval(output[0])  # force this layer's matmuls to execute
            self._tl.record(self._idx, self._load_start, self._load_end, time.perf_counter())
            self._seen.add(self._idx)
        self._idx = None
        return output


@contextmanager
def capture_layer_timeline() -> Iterator[LayerTimeline]:
    """Record real per-layer load/compute while AirLLM's MLX backend generates.

    Yields a :class:`LayerTimeline`. On non-macOS / no-MLX hosts it yields an
    empty timeline and patches nothing, so callers stay portable.
    """
    timeline = LayerTimeline()
    try:
        import mlx.core as mx
        from airllm.airllm_llama_mlx import TransformerBlock
        from airllm.persist.mlx_model_persister import MlxModelPersister
    except Exception:  # pragma: no cover - non-mac / mlx missing
        yield timeline
        return

    cap = _Capture(timeline, mx.eval)
    orig_load = MlxModelPersister.load_model
    orig_call = TransformerBlock.__call__

    def timed_load(self, layer_name, path):  # noqa: ANN001, ANN202
        return cap.on_load(layer_name, lambda: orig_load(self, layer_name, path))

    def timed_call(self, x, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        return cap.on_forward(orig_call(self, x, *args, **kwargs))

    MlxModelPersister.load_model = timed_load
    TransformerBlock.__call__ = timed_call
    try:
        yield timeline
    finally:
        MlxModelPersister.load_model = orig_load
        TransformerBlock.__call__ = orig_call


def write_timeline(dicts: list[dict], path: Path) -> None:
    """Persist per-layer timeline dicts to ``path`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dicts, indent=2))


def read_timeline(path: Path) -> list[dict]:
    """Load per-layer timeline dicts from ``path``; return [] if absent/invalid."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else []


def io_fraction_of(dicts: list[dict]) -> float:
    """Share of total per-layer wall-time spent on disk load (vs forward compute)."""
    total = sum(d["load_ms"] + d["compute_ms"] for d in dicts)
    return sum(d["load_ms"] for d in dicts) / total if total else 0.0
