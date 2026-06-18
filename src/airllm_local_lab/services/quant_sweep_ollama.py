"""Task 1: Ollama-backed precision sweep — real TTFT + TPOT via /api/generate."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import time
from pathlib import Path

from airllm_local_lab.sdk.model_loader.ollama_backend import OllamaBackend
from airllm_local_lab.sdk.quality.rater import rate
from airllm_local_lab.shared.config import load_config
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)
RESULTS = Path(__file__).resolve().parents[3] / "results"

# llama3.2:1b quant variants — confirmed locally available
SWEEP_TAGS = [
    ("llama3.2:1b-instruct-q8_0", "Q8_0", "llama3.2"),
    ("llama3.2:1b-instruct-q4_K_M", "Q4_K_M", "llama3.2"),
    ("llama3.2:1b-instruct-q2_K", "Q2_K", "llama3.2"),
]
REPS = 3
MAX_TOKENS = 20


def _ensure_pulled(model_tag: str) -> None:
    res = subprocess.run(["ollama", "show", model_tag], capture_output=True, timeout=10)
    if res.returncode != 0:
        log.info("Pulling %s …", model_tag)
        subprocess.run(["ollama", "pull", model_tag], check=True, timeout=600)


def _run_rep(backend: OllamaBackend, prompt: str, rep: int) -> dict:
    cache_state = "cold" if rep == 0 else "warm"
    t0 = time.perf_counter()
    try:
        m = backend.generate_with_metrics(prompt, max_new_tokens=MAX_TOKENS)
        wall_s = time.perf_counter() - t0
        quality = rate(m["text"], prompt)
        return {
            "rep": rep + 1,
            "cache_state": cache_state,
            "text": m["text"][:120],
            "eval_count": m["eval_count"],
            "prompt_eval_count": m["prompt_eval_count"],
            "ttft_s": round(m["prompt_eval_s"], 4),
            "tpot_s": round(m["tpot_s"], 6),
            "eval_s": round(m["eval_s"], 4),
            "total_s": round(m["total_s"], 4),
            "wall_s": round(wall_s, 4),
            "throughput_tps": round(m["throughput_tps"], 3),
            "quality_normalised": round(quality.normalised, 3) if quality else 0.0,
            "status": "ok",
            "error": None,
        }
    except Exception as exc:
        log.warning("Rep %d failed: %s", rep + 1, exc)
        return {"rep": rep + 1, "cache_state": cache_state, "status": "error", "error": str(exc)[:200]}


def run_one_tag(model_tag: str, label: str, base: str, prompt: str) -> dict:
    """Run REPS repetitions for one model tag; return aggregated result row."""
    _ensure_pulled(model_tag)
    backend = OllamaBackend(model_tag=model_tag, max_new_tokens=MAX_TOKENS)
    backend.load()

    reps_data: list[dict] = []
    for i in range(REPS):
        reps_data.append(_run_rep(backend, prompt, i))
        log.info(
            "  %s rep %d → tpot=%.1f ms tput=%.1f tok/s",
            model_tag,
            i + 1,
            reps_data[-1].get("tpot_s", 0) * 1000,
            reps_data[-1].get("throughput_tps", 0),
        )

    ok_reps = [r for r in reps_data if r.get("status") == "ok"]
    if not ok_reps:
        return {"model_tag": model_tag, "precision": label, "status": "error", "reps_raw": reps_data}

    size_mb = backend.get_model_size_mb()

    ttfts = [r["ttft_s"] for r in ok_reps]
    tpots = [r["tpot_s"] for r in ok_reps]
    tputs = [r["throughput_tps"] for r in ok_reps]
    quals = [r["quality_normalised"] for r in ok_reps]

    return {
        "backend": "ollama",
        "model_id": base,
        "model_tag": model_tag,
        "precision": label,
        "status": "ok",
        "error": None,
        "output": ok_reps[0].get("text", "")[:200],
        "num_tokens": ok_reps[0].get("eval_count", 0),
        "reps": len(ok_reps),
        "ttft_median_s": round(statistics.median(ttfts), 4),
        "ttft_iqr_s": round(max(ttfts) - min(ttfts), 4),
        "tpot_median_s": round(statistics.median(tpots), 6),
        "tpot_iqr_s": round(max(tpots) - min(tpots), 6),
        "throughput_median_tps": round(statistics.median(tputs), 3),
        "throughput_iqr_tps": round(max(tputs) - min(tputs), 3),
        "quality_normalised": round(statistics.median(quals), 3),
        "peak_ram_mb": round(size_mb, 1),
        "shard_size_gb": round(size_mb / 1024, 3),
        "total_s": round(statistics.median([r.get("total_s", 0) for r in ok_reps]), 3),
        "reps_raw": reps_data,
    }


def main() -> None:
    cfg = load_config()
    RESULTS.mkdir(parents=True, exist_ok=True)
    prompt = cfg.benchmark.prompt_set[0]

    rows: list[dict] = []
    for tag, label, base in SWEEP_TAGS:
        log.info("=== Ollama quant sweep: %s (%s) ===", tag, label)
        row = run_one_tag(tag, label, base, prompt)
        rows.append(row)
        log.info(
            "  %s → ttft=%.3f s tpot=%.1f ms tput=%.1f tok/s qual=%.3f",
            label,
            row.get("ttft_median_s", 0),
            row.get("tpot_median_s", 0) * 1000,
            row.get("throughput_median_tps", 0),
            row.get("quality_normalised", 0),
        )

    (RESULTS / "quant_sweep_ollama.json").write_text(json.dumps(rows, indent=2))

    if rows:
        csv_keys = [k for k in rows[0] if k != "reps_raw"]
        with open(RESULTS / "quant_sweep_ollama.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=csv_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    log.info("Ollama quant sweep → results/quant_sweep_ollama.json + .csv")
