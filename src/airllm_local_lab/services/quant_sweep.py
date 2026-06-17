"""Phase 3: Quantization sweep — FP16, 8bit, 4bit via AirLLM."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from airllm_local_lab.sdk.metrics.memory import MemorySampler
from airllm_local_lab.sdk.model_loader.airllm_backend import AirLLMBackend
from airllm_local_lab.sdk.quality.rater import rate
from airllm_local_lab.shared.config import load_config
from airllm_local_lab.shared.gatekeeper import Gatekeeper
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)
RESULTS = Path(__file__).resolve().parents[3] / "results"

PRECISION_MAP = {
    "fp16": None,
    "8bit": "4bit",
    "4bit": "4bit",
}


def _shard_size_gb(shards_path: str, model_id: str) -> float:
    model_slug = model_id.replace("/", "--")
    shard_dir = Path(shards_path) / model_slug
    if not shard_dir.exists():
        return 0.0
    total = sum(f.stat().st_size for f in shard_dir.rglob("*") if f.is_file())
    return round(total / 1e9, 2)


def run_precision(
    model_id: str,
    precision: str,
    shards_path: str,
    token: str | None,
    prompt: str,
    max_new_tokens: int = 32,
) -> dict:
    compression = PRECISION_MAP.get(precision)
    if precision == "8bit":
        compression = "8bit"
    elif precision == "4bit":
        compression = "4bit"
    else:
        compression = None

    log.info("Sweep: %s precision=%s compression=%s", model_id, precision, compression)
    backend = AirLLMBackend(
        model_id=model_id,
        shards_path=shards_path,
        compression=compression,
        token=token,
    )
    sampler = MemorySampler()
    sampler.start()
    t0 = time.perf_counter()

    error_msg = None
    output_text = ""
    num_tokens = 0
    try:
        backend.load()
        result = backend.generate(prompt, max_new_tokens=max_new_tokens)
        output_text = result.text
        num_tokens = result.num_tokens
    except Exception as exc:
        error_msg = str(exc)[:300]
        log.warning("Precision %s failed: %s", precision, error_msg)
    finally:
        total_s = time.perf_counter() - t0
        mem = sampler.stop()
        backend.unload()

    quality = rate(output_text, prompt) if output_text else None
    shard_gb = _shard_size_gb(shards_path, model_id)

    return {
        "model_id": model_id,
        "precision": precision,
        "compression": compression,
        "status": "error" if error_msg else "ok",
        "error": error_msg,
        "output": output_text[:200],
        "num_tokens": num_tokens,
        "total_s": round(total_s, 3),
        "peak_ram_mb": round(mem.peak_ram_mb, 1),
        "shard_size_gb": shard_gb,
        "throughput_tps": round(num_tokens / total_s, 4) if total_s and num_tokens else 0,
        "quality_normalised": round(quality.normalised, 3) if quality else 0,
    }


def main() -> None:
    cfg = load_config()
    gk = Gatekeeper()
    RESULTS.mkdir(parents=True, exist_ok=True)

    shards_path = str(Path(cfg.model.layer_shards_path).expanduser())
    model_id = cfg.model.sweep_model_id
    prompt = cfg.benchmark.prompt_set[0]

    rows: list[dict] = []
    for precision in ["fp16", "8bit", "4bit"]:
        row = run_precision(model_id, precision, shards_path, gk.hf_token(), prompt, cfg.model.max_new_tokens)
        rows.append(row)
        log.info(
            "  %s → peak_ram=%.0f MB shard=%.1f GB tok/s=%.3f",
            precision,
            row["peak_ram_mb"],
            row["shard_size_gb"],
            row["throughput_tps"],
        )

    (RESULTS / "quant_sweep.json").write_text(json.dumps(rows, indent=2))

    with open(RESULTS / "quant_sweep.csv", "w", newline="") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    log.info("Quant sweep complete → results/quant_sweep.json + .csv")
