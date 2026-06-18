"""Phase 7: Assemble all figures/tables into README.md report."""

from __future__ import annotations

import json
from pathlib import Path

from airllm_local_lab.sdk.viz.tables import hardware_table, precision_sweep_table
from airllm_local_lab.services._report_architecture import (
    section_architecture,
    section_quality_gates,
)
from airllm_local_lab.services._report_content import (
    section_contributing,
    section_research_questions,
    section_run_instructions,
)
from airllm_local_lab.services._report_extras import (
    section_kpi_scorecard,
    section_prompt_log,
    section_raw_benchmark,
    section_token_costs,
)
from airllm_local_lab.services._report_planning import section_planning_docs
from airllm_local_lab.services._report_sections import (
    _HW,
    section_airllm,
    section_baseline,
    section_header,
    section_models,
)
from airllm_local_lab.services._report_theory import section_theory_iso
from airllm_local_lab.shared.logging import get_logger
from airllm_local_lab.shared.version import __version__

log = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"


def _load_json(path: Path) -> dict | list | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def _fig(name: str, caption: str) -> str:
    p = ASSETS / name
    if p.exists():
        return f"![{caption}](assets/{name})\n*{caption}*\n"
    return f"*Figure not yet generated: {name}*\n"


def main() -> None:
    baseline = _load_json(RESULTS / "baseline.json") or {}
    demo = _load_json(RESULTS / "airllm_demo.json") or {}
    economics = _load_json(RESULTS / "economics.json") or {}
    summary_rows: list[dict] = _load_json(RESULTS / "benchmark_summary.json") or []  # type: ignore[assignment]
    raw_rows: list[dict] = _load_json(RESULTS / "benchmark_raw.json") or []  # type: ignore[assignment]
    e1_data: dict = _load_json(RESULTS / "extension_e1.json") or {}  # type: ignore[assignment]
    e3_data: dict = _load_json(RESULTS / "extension_e3.json") or {}  # type: ignore[assignment]
    ok_rows = [r for r in summary_rows if isinstance(r, dict) and r.get("status") == "ok"]
    infeasible_rows = [r for r in summary_rows if isinstance(r, dict) and r.get("status") != "ok"]

    hw_md = hardware_table(_HW)
    sweep_md = precision_sweep_table(ok_rows) if ok_rows else "*Sweep not yet run.*"
    if infeasible_rows:
        reasons = "\n".join(
            f"- **{r.get('precision', '?')}:** {r.get('reason', r.get('error', 'failed'))} "
            f"*(negative result — documented)*"
            for r in infeasible_rows
        )
        sweep_md += (
            "\n\n**Quantization negative results (8bit / 4bit):**\n\n"
            + reasons
            + "\n\n> **macOS ARM constraint:** `bitsandbytes` (required by AirLLM for sub-FP16 "
            "quantization) is CUDA-only and does not support Apple Silicon. "
            "This is a hardware-platform negative result, not a code bug. "
            "On a Linux/CUDA system, 8bit would yield ~2× smaller shards and ~2× faster I/O; "
            "4bit ~4× smaller shards with minor quality loss. "
            "The theoretical trade-off is documented in §9.3.\n\n"
            "> **Note on TPOT = 0.0:** The AirLLM MLX backend on macOS generates all output tokens "
            "in a single batched call (not token-by-token autoregressive decoding). "
            "Per-token inter-token latency (TPOT/ITL) is therefore not separately measurable — "
            "the entire generation time is captured in TTFT. "
            "Theoretically, TPOT on AirLLM would be dominated by NVMe shard re-read latency "
            "per decoder layer (~18–20 layers/second), as analysed in §9.1."
        )
    assumptions_file = ASSETS / "economics_assumptions.md"
    assumptions_md = (
        assumptions_file.read_text() if assumptions_file.exists() else "*Economics not yet run.*"
    )
    crossover = economics.get("crossover_tokens")
    crossover_str = f"{crossover:,.0f}" if crossover else "N/A"

    fig_captions = [
        ("F1_memory_footprint.png", "F1 — Peak RAM and shard size vs precision"),
        ("F2_latency.png", "F2 — TTFT and TPOT vs precision"),
        ("F3_throughput.png", "F3 — Throughput vs precision"),
        ("F4_quality_vs_memory.png", "F4 — Quality vs memory trade-off"),
        ("F5_layer_timeline.png", "F5 — Per-layer load vs compute timeline"),
        ("F6_page_cache_warmup.png", "F6 — Cold to warm page-cache speedup"),
        ("F7_io_location.png", "F7 — Shard-location I/O sensitivity (Extension E1)"),
        ("E_breakeven.png", "Break-even: On-Prem vs Managed API vs Cloud GPU"),
    ]
    figs = {n: _fig(n, c) for n, c in fig_captions}

    sections = [
        section_header(__version__),
        section_planning_docs(),
        f"## 1. Hardware\n\n{hw_md}\n\n**Critical constraint:** 18 GB unified RAM vs 26 GB for "
        "OPT-13b FP16 → direct load fails. AirLLM's layer-streaming is the only feasible path.",
        section_models(),
        section_baseline(baseline.get("sanity", {}), baseline.get("direct_load", {})),
        section_research_questions(),
        section_airllm(demo),
        f"## 6. Quantization Sweep\n\n{sweep_md}\n\n"
        f"{figs['F1_memory_footprint.png']}{figs['F2_latency.png']}"
        f"{figs['F3_throughput.png']}{figs['F4_quality_vs_memory.png']}",
        "## 7. Benchmarking\n\n"
        + section_raw_benchmark(raw_rows)
        + f"\n{figs['F5_layer_timeline.png']}{figs['F6_page_cache_warmup.png']}"
        + f"{figs['F7_io_location.png']}",
        f"## 8. Economic Analysis\n\n**Break-even:** ~{crossover_str} tokens/month\n\n"
        f"{figs['E_breakeven.png']}\n\n### Assumptions\n{assumptions_md}"
        + section_token_costs(),
        section_theory_iso(
            figs["F7_io_location.png"], figs["F6_page_cache_warmup.png"], e1_data, e3_data
        ),
        section_architecture(),
        section_quality_gates(),
        section_kpi_scorecard(),
        section_prompt_log(),
        section_run_instructions(),
        section_contributing(),
        f"*Generated by `uv run report` · v{__version__} · 2026-06-18*",
    ]

    (ROOT / "README.md").write_text("\n\n---\n\n".join(sections))
    log.info("README.md report assembled (v%s)", __version__)
