"""Phase 4: Full benchmark pipeline — ≥3 reps, cold/warm, all metrics."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import torch

from airllm_local_lab.sdk.model_loader.airllm_backend import AirLLMBackend
from airllm_local_lab.sdk.viz import plots
from airllm_local_lab.services._benchmark_helpers import _run_single, _summarise_reps
from airllm_local_lab.shared.config import load_config
from airllm_local_lab.shared.gatekeeper import Gatekeeper
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)
RESULTS = Path(__file__).resolve().parents[3] / "results"
ASSETS = Path(__file__).resolve().parents[3] / "assets"


def _set_determinism(seed: int, num_threads: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)


def run_matrix(
    model_id: str,
    precisions: list[str],
    shards_path: str,
    token: str | None,
    prompt: str,
    max_new_tokens: int,
    reps: int,
    seed: int,
    num_threads: int,
) -> tuple[list[dict], list[dict]]:
    _set_determinism(seed, num_threads)
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict] = []
    summary_rows: list[dict] = []

    for precision in precisions:
        compression = {"8bit": "8bit", "4bit": "4bit"}.get(precision)
        backend = AirLLMBackend(model_id=model_id, shards_path=shards_path, compression=compression, token=token)

        try:
            backend.load()
        except Exception as exc:
            log.warning("Failed to load %s @ %s: %s", model_id, precision, exc)
            summary_rows.append({"precision": precision, "status": "infeasible", "reason": str(exc)[:200]})
            continue

        rep_rows: list[dict] = []
        for rep in range(1, reps + 1):
            cache_state = "cold" if rep == 1 else "warm"
            row = _run_single(backend, prompt, max_new_tokens, cache_state)
            row.update({"model_id": model_id, "precision": precision, "rep": rep})
            raw_rows.append(row)
            rep_rows.append(row)
            log.info(
                "  %s rep %d/%d — %.2f s  %.3f tok/s  ram=%.0f MB",
                precision,
                rep,
                reps,
                row["total_s"],
                row["throughput_tps"],
                row["peak_ram_mb"],
            )

        backend.unload()
        summary = _summarise_reps(rep_rows)
        summary["precision"] = precision
        summary["model_id"] = model_id
        summary["status"] = "ok"
        summary_rows.append(summary)

    return raw_rows, summary_rows


def main() -> None:
    cfg = load_config()
    gk = Gatekeeper()
    shards_path = str(Path(cfg.model.layer_shards_path).expanduser())

    raw, summary = run_matrix(
        model_id=cfg.model.sweep_model_id,
        precisions=["fp16", "8bit", "4bit"],
        shards_path=shards_path,
        token=gk.hf_token(),
        prompt=cfg.benchmark.prompt_set[0],
        max_new_tokens=cfg.model.max_new_tokens,
        reps=cfg.benchmark.repetitions,
        seed=cfg.benchmark.seed,
        num_threads=cfg.benchmark.num_threads,
    )

    (RESULTS / "benchmark_raw.json").write_text(json.dumps(raw, indent=2))
    (RESULTS / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))

    if raw:
        with open(RESULTS / "benchmark_raw.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=raw[0].keys())
            writer.writeheader()
            writer.writerows(raw)

    log.info("Benchmark complete. Generating figures...")
    ok_rows = [r for r in summary if r.get("status") == "ok"]
    if ok_rows:
        plots.f1_memory_footprint(ok_rows)
        plots.f2_latency(ok_rows)
        plots.f3_throughput(ok_rows)
        plots.f4_quality_vs_memory(ok_rows)
    log.info("Figures saved to assets/")
