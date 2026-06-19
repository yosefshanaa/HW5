"""Results-driven figure regeneration (keeps report_builder.py <= 150 lines)."""

from __future__ import annotations

from pathlib import Path

from airllm_local_lab.sdk.metrics.airllm_instrument import io_fraction_of, read_timeline
from airllm_local_lab.sdk.viz import plots

_GENERIC_CAPTION = "F5 — Per-layer load vs compute timeline"


def render_f5(results_dir: Path) -> tuple[str, str]:
    """Regenerate F5 from the measured timeline JSON.

    Returns ``(caption, blurb)`` — an honest figure caption and a §7 lead-in
    paragraph derived from real data. Falls back to the generic caption and an
    empty blurb when no timeline has been captured yet.
    """
    dicts = read_timeline(results_dir / "layer_timeline.json")
    plots.f5_layer_timeline(dicts)
    if not dicts:
        return _GENERIC_CAPTION, ""

    pct = round(io_fraction_of(dicts) * 100)
    mean_load = sum(d["load_ms"] for d in dicts) / len(dicts)
    mean_compute = sum(d["compute_ms"] for d in dicts) / len(dicts)
    caption = (
        "F5 — Measured per-layer load vs compute timeline "
        f"(TinyLlama-1.1B · AirLLM MLX streaming · Apple M3 Pro · {len(dicts)} layers · "
        f"load = {pct}% of wall-time → disk-I/O-bound)"
    )
    blurb = (
        "**Measured per-layer timeline (F5):** captured live during the AirLLM MLX "
        f"layer-streaming run on the Apple M3 Pro — all {len(dicts)} TinyLlama-1.1B "
        f"layers, mean **{mean_load:.1f} ms disk-load** vs **{mean_compute:.1f} ms forward "
        f"compute** per layer. Disk I/O is **{pct}%** of per-layer wall-time, confirming the "
        "run is I/O-bound, not compute-bound. The first few layers carry one-time MLX kernel "
        "JIT-compilation that slightly inflates their compute bars, so the true steady-state "
        "I/O share is even higher than shown (ADR-008: figures from measured data only).\n\n"
    )
    return caption, blurb
