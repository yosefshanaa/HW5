"""Phase 2: AirLLM integration — giant model via layer-streaming on CPU/MPS."""

from __future__ import annotations

import json
import time
from pathlib import Path

from airllm_local_lab.sdk.metrics.memory import MemorySampler
from airllm_local_lab.sdk.model_loader.airllm_backend import AirLLMBackend
from airllm_local_lab.shared.config import load_config
from airllm_local_lab.shared.gatekeeper import Gatekeeper
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)
RESULTS = Path(__file__).resolve().parents[3] / "results"


def run_airllm_demo(
    model_id: str,
    shards_path: str,
    token: str | None,
    prompt: str = "Explain what a transformer is in one sentence.",
    max_new_tokens: int = 32,
    compression: str | None = None,
) -> dict:
    log.info("=== AirLLM demo: %s (compression=%s) ===", model_id, compression)
    backend = AirLLMBackend(
        model_id=model_id,
        shards_path=shards_path,
        compression=compression,
        token=token,
    )
    sampler = MemorySampler()
    sampler.start()
    t0 = time.perf_counter()
    backend.load()
    result = backend.generate(prompt, max_new_tokens=max_new_tokens)
    total_s = time.perf_counter() - t0
    mem = sampler.stop()
    backend.unload()

    record = {
        "model_id": model_id,
        "compression": compression,
        "prompt": prompt,
        "output": result.text,
        "num_tokens": result.num_tokens,
        "total_s": round(total_s, 3),
        "peak_ram_mb": round(mem.peak_ram_mb, 1),
        "peak_vram_mb": round(mem.peak_vram_mb, 1),
        "throughput_tps": round(result.num_tokens / total_s, 4) if total_s else 0,
    }
    log.info("AirLLM demo complete: %d tokens in %.1f s", result.num_tokens, total_s)
    return record


def main() -> None:
    cfg = load_config()
    gk = Gatekeeper()
    RESULTS.mkdir(parents=True, exist_ok=True)

    shards_path = str(Path(cfg.model.layer_shards_path).expanduser())
    record = run_airllm_demo(
        model_id=cfg.model.model_id,
        shards_path=shards_path,
        token=gk.hf_token(),
        max_new_tokens=cfg.model.max_new_tokens,
    )

    out_file = RESULTS / "airllm_demo.json"
    out_file.write_text(json.dumps(record, indent=2))
    log.info("AirLLM demo written to %s", out_file)
    print("\nOutput:", record["output"][:300])
