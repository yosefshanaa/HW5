"""All figure generators — F1 through F7 + break-even chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from airllm_local_lab.sdk.viz._plots_extra import _breakeven_impl, _f6_impl, _f7_impl

matplotlib.use("Agg")
ASSETS = Path(__file__).resolve().parents[5] / "assets"


def _save(fig: plt.Figure, name: str) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def f1_memory_footprint(summary_rows: list[dict]) -> Path:
    precisions = [r["precision"] for r in summary_rows]
    peak_ram = [r.get("peak_ram_mb", 0) / 1024 for r in summary_rows]
    shard_gb = [r.get("shard_size_gb", 0) for r in summary_rows]

    x = range(len(precisions))
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.35
    ax.bar([i - w / 2 for i in x], peak_ram, w, label="Peak RAM (GiB)")
    ax.bar([i + w / 2 for i in x], shard_gb, w, label="Shard size (GB)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(precisions)
    ax.set_ylabel("GiB / GB")
    ax.set_title("F1 — Memory footprint vs precision")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "F1_memory_footprint.png")


def f2_latency(summary_rows: list[dict]) -> Path:
    precisions = [r["precision"] for r in summary_rows]
    ttft = [r.get("ttft_median_s", 0) for r in summary_rows]
    tpot = [r.get("tpot_median_s", 0) for r in summary_rows]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(precisions, ttft, "b-o", label="TTFT (s)")
    ax2.plot(precisions, tpot, "r-s", label="TPOT (s)")
    ax1.set_ylabel("TTFT (s)", color="b")
    ax2.set_ylabel("TPOT (s)", color="r")
    ax1.set_title("F2 — Latency vs precision")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    fig.tight_layout()
    return _save(fig, "F2_latency.png")


def f3_throughput(summary_rows: list[dict]) -> Path:
    precisions = [r["precision"] for r in summary_rows]
    tps = [r.get("throughput_median_tps", 0) for r in summary_rows]
    err = [r.get("throughput_iqr_tps", 0) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(precisions, tps, yerr=err, capsize=5, color="steelblue")
    ax.set_ylabel("Tokens / second")
    ax.set_title("F3 — Throughput vs precision")
    fig.tight_layout()
    return _save(fig, "F3_throughput.png")


def f4_quality_vs_memory(summary_rows: list[dict]) -> Path:
    precisions = [r["precision"] for r in summary_rows]
    quality = [r.get("quality_normalised", 0) for r in summary_rows]
    mem_frac = [r.get("memory_fraction_of_fp16", 1.0) for r in summary_rows]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(precisions, quality, "g-o", label="Quality (0–1)")
    ax2.plot(precisions, mem_frac, "m-s", label="RAM / FP16 RAM")
    ax1.set_ylabel("Quality score", color="g")
    ax2.set_ylabel("Relative memory", color="m")
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(0, 1.1)
    ax1.set_title("F4 — Quality vs memory trade-off")
    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2)
    fig.tight_layout()
    return _save(fig, "F4_quality_vs_memory.png")


def f5_layer_timeline(layer_dicts: list[dict]) -> Path:
    if not layer_dicts:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No timeline data", ha="center", va="center")
        return _save(fig, "F5_layer_timeline.png")
    layers = [d["layer"] for d in layer_dicts]
    load_ms = [d["load_ms"] for d in layer_dicts]
    compute_ms = [d["compute_ms"] for d in layer_dicts]

    fig, ax = plt.subplots(figsize=(max(10, len(layers) // 2), 4))
    ax.bar(layers, load_ms, label="Load (ms)", color="tomato")
    ax.bar(layers, compute_ms, bottom=load_ms, label="Compute (ms)", color="steelblue")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Time (ms)")
    ax.set_title("F5 — Per-layer timeline (I/O-bound pattern)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "F5_layer_timeline.png")


def f6_page_cache_warmup(cold_s: list[float], warm_s: list[float]) -> Path:
    return _f6_impl(cold_s, warm_s, _save)


def f7_io_location(location_results: dict[str, float]) -> Path:
    return _f7_impl(location_results, _save)


def breakeven_chart(volumes: list[float], curves: dict[str, list[float]], crossover: float | None) -> Path:
    return _breakeven_impl(volumes, curves, crossover, _save)
