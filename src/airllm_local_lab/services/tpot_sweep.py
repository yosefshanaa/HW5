"""Task 3: AirLLM empirical TPOT — run at token-counts 1,2,4,8; diff wall times."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from airllm_local_lab.sdk.model_loader.airllm_backend import AirLLMBackend
from airllm_local_lab.shared.config import load_config
from airllm_local_lab.shared.gatekeeper import Gatekeeper
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)
RESULTS = Path(__file__).resolve().parents[3] / "results"

TOKEN_COUNTS = [1, 2, 4, 8]
REPS = 3
PROMPT = "Explain what a transformer is."


def _run_one(backend: AirLLMBackend, max_new_tokens: int) -> float:
    """Run one generation, return wall-clock seconds."""
    t0 = time.perf_counter()
    backend.generate(PROMPT, max_new_tokens=max_new_tokens)
    return time.perf_counter() - t0


def _median_time(backend: AirLLMBackend, max_new_tokens: int, reps: int = REPS) -> dict:
    times = []
    for i in range(reps):
        t = _run_one(backend, max_new_tokens)
        times.append(t)
        log.info("  n_tokens=%d rep=%d time=%.2f s", max_new_tokens, i + 1, t)
    return {
        "max_new_tokens": max_new_tokens,
        "times_s": [round(t, 3) for t in times],
        "median_s": round(statistics.median(times), 3),
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
    }


def _fit_tpot_ms(data: list[dict]) -> dict:
    """Estimate TPOT via linear fit: time(n) = base + tpot * n."""
    xs = [d["max_new_tokens"] for d in data]
    ys = [d["median_s"] for d in data]
    n = len(xs)
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    ss_xy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys, strict=True))
    ss_xx = sum((x - x_bar) ** 2 for x in xs)
    tpot_s = ss_xy / ss_xx if ss_xx else 0.0
    base_s = y_bar - tpot_s * x_bar
    return {
        "tpot_ms_linear_fit": round(tpot_s * 1000, 2),
        "base_overhead_s": round(base_s, 3),
        "note": (
            "TPOT estimated by linear fit: time(n) = base_overhead + tpot*n. "
            "AirLLM MLX batches, so this measures the per-token marginal cost "
            "of layer-streaming on NVMe (I/O-bound)."
        ),
    }


def main() -> None:
    cfg = load_config()
    gk = Gatekeeper()
    RESULTS.mkdir(parents=True, exist_ok=True)

    shards_path = str(Path(cfg.model.layer_shards_path).expanduser())
    model_id = cfg.model.sweep_model_id

    log.info("=== AirLLM TPOT sweep: %s at token counts %s ===", model_id, TOKEN_COUNTS)
    backend = AirLLMBackend(model_id=model_id, shards_path=shards_path, token=gk.hf_token())

    try:
        backend.load()
    except Exception as exc:
        log.error("Failed to load model: %s", exc)
        (RESULTS / "tpot_sweep.json").write_text(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return

    points: list[dict] = []
    for n in TOKEN_COUNTS:
        pt = _median_time(backend, n)
        points.append(pt)

    backend.unload()

    fit = _fit_tpot_ms(points)
    result = {
        "model_id": model_id,
        "backend": "airllm_mlx",
        "prompt": PROMPT,
        "token_counts": TOKEN_COUNTS,
        "measurements": points,
        **fit,
    }
    (RESULTS / "tpot_sweep.json").write_text(json.dumps(result, indent=2))
    log.info("TPOT sweep complete — estimated TPOT = %.1f ms (linear fit)", fit["tpot_ms_linear_fit"])
    log.info("Results → results/tpot_sweep.json")
