"""F6, F7, and break-even chart implementations (save_fn-threaded)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch


def _f6_impl(cold_s: list[float], warm_s: list[float], save_fn: Callable) -> Path:
    runs = list(range(1, len(cold_s) + len(warm_s) + 1))
    all_times = cold_s + warm_s
    labels = ["cold"] * len(cold_s) + ["warm"] * len(warm_s)
    colors = ["tomato" if lbl == "cold" else "steelblue" for lbl in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(runs, all_times, color=colors)
    ax.set_xlabel("Run number")
    ax.set_ylabel("Total runtime (s)")
    ax.set_title("F6 — Page-cache warmup: cold vs warm runs")
    legend = [Patch(color="tomato", label="cold"), Patch(color="steelblue", label="warm")]
    ax.legend(handles=legend)
    fig.tight_layout()
    return save_fn(fig, "F6_page_cache_warmup.png")


def _f7_impl(location_results: dict[str, float], save_fn: Callable) -> Path:
    locations = list(location_results.keys())
    times = list(location_results.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(locations, times, color=["steelblue", "darkorange"])
    ax.set_ylabel("Total runtime (s)")
    ax.set_title("F7 — Shard-location I/O sensitivity (Extension E1)")
    fig.tight_layout()
    return save_fn(fig, "F7_io_location.png")


def _breakeven_impl(
    volumes: list[float],
    curves: dict[str, list[float]],
    crossover: float | None,
    save_fn: Callable,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"onprem": "steelblue", "api": "tomato", "cloud_gpu": "green"}
    labels = {"onprem": "On-Prem (this laptop)", "api": "Managed API", "cloud_gpu": "Cloud GPU"}
    for key, costs in curves.items():
        ax.plot(volumes, costs, label=labels.get(key, key), color=colors.get(key, "grey"))
    if crossover is not None:
        ax.axvline(crossover, linestyle="--", color="gray", label=f"Break-even ≈ {crossover:,.0f} tok/mo")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    ax.set_xlabel("Monthly token volume")
    ax.set_ylabel("Monthly cost (USD)")
    ax.set_title("Economic Break-Even: On-Prem vs Managed API")
    ax.legend()
    fig.tight_layout()
    return save_fn(fig, "E_breakeven.png")
