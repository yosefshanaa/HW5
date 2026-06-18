"""Generate Markdown tables from results data."""

from __future__ import annotations

from pathlib import Path

ASSETS = Path(__file__).resolve().parents[4] / "assets"


def _md_table(headers: list[str], rows: list[list]) -> str:
    sep = " | ".join(["---"] * len(headers))
    header_row = " | ".join(headers)
    data_rows = [" | ".join(str(c) for c in row) for row in rows]
    return f"| {header_row} |\n| {sep} |\n" + "\n".join(f"| {r} |" for r in data_rows)


def precision_sweep_table(summary_rows: list[dict]) -> str:
    headers = ["Precision", "Peak RAM (GiB)", "Shard (GB)", "TTFT (s)", "TPOT (s)", "Throughput (tok/s)", "Quality"]
    rows = [
        [
            r.get("precision", ""),
            f"{r.get('peak_ram_mb', 0) / 1024:.2f}",
            f"{r.get('shard_size_gb', 0):.1f}",
            f"{r.get('ttft_median_s', 0):.2f}",
            f"{r.get('tpot_median_s', 0):.3f}",
            f"{r.get('throughput_median_tps', 0):.3f}",
            f"{r.get('quality_normalised', 0):.2f}",
        ]
        for r in summary_rows
    ]
    return _md_table(headers, rows)


def economics_assumptions_table(assumptions: list[dict]) -> str:
    headers = ["Parameter", "Value", "Source / Date"]
    rows = [[a["parameter"], a["value"], a.get("source", "—")] for a in assumptions]
    return _md_table(headers, rows)


def hardware_table(hw: dict) -> str:
    headers = ["Component", "Specification", "Implication"]
    rows = [[k, v["spec"], v["implication"]] for k, v in hw.items()]
    return _md_table(headers, rows)


def ollama_quant_table(ollama_rows: list[dict]) -> str:
    if not ollama_rows:
        return "*Ollama quant sweep not yet run.*"
    lines = ["| Precision | Backend | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | RAM (MB) | Quality |"]
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in ollama_rows:
        lines.append(
            f"| {r['precision']} | Ollama/GGUF | {r.get('ttft_median_s', 0) * 1000:.0f} "
            f"| {r.get('tpot_median_s', 0) * 1000:.1f} | {r.get('throughput_median_tps', 0):.1f} "
            f"| {r.get('peak_ram_mb', 0):.0f} | {r.get('quality_normalised', 0):.3f} |"
        )
    return "\n".join(lines)
