"""Extension E1 — Shard-location I/O sensitivity study.

Benchmarks identical AirLLM runs with shards on:
  (a) ~/airllm_cache  — internal NVMe SSD (APFS)
  (b) /tmp/airllm_cache — often on same disk but avoids Spotlight indexing; on macOS
      /tmp is a RAM-backed tmpfs on some systems providing an upper-bound comparison.

Records total_s per location and emits F7.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from airllm_local_lab.sdk.model_loader.airllm_backend import AirLLMBackend
from airllm_local_lab.sdk.viz.plots import f7_io_location
from airllm_local_lab.shared.config import load_config
from airllm_local_lab.shared.gatekeeper import Gatekeeper
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)
RESULTS = Path(__file__).resolve().parents[4] / "results"

LOCATIONS = {
    "internal_ssd": "~/airllm_cache",
    "tmp": "/tmp/airllm_cache",
}


def _run_location(model_id: str, shards_path: str, token: str | None, prompt: str, max_new_tokens: int) -> float:
    backend = AirLLMBackend(model_id=model_id, shards_path=shards_path, token=token)
    t0 = time.perf_counter()
    backend.load()
    backend.generate(prompt, max_new_tokens=max_new_tokens)
    elapsed = time.perf_counter() - t0
    backend.unload()
    return round(elapsed, 3)


def main() -> None:
    cfg = load_config()
    gk = Gatekeeper()
    RESULTS.mkdir(parents=True, exist_ok=True)

    model_id = cfg.model.sweep_model_id
    prompt = cfg.benchmark.prompt_set[0]
    results: dict[str, float] = {}

    for label, raw_path in LOCATIONS.items():
        resolved = str(Path(raw_path).expanduser())
        log.info("E1: running on %s → %s", label, resolved)
        try:
            t = _run_location(model_id, resolved, gk.hf_token(), prompt, cfg.model.max_new_tokens)
            results[label] = t
            log.info("  %s: %.2f s", label, t)
        except Exception as exc:
            log.warning("  %s failed: %s", label, exc)
            results[label] = -1.0

    (RESULTS / "extension_e1.json").write_text(json.dumps(results, indent=2))
    valid = {k: v for k, v in results.items() if v > 0}
    if valid:
        f7_io_location(valid)
    log.info("E1 complete → results/extension_e1.json + assets/F7_io_location.png")
