"""Extension E3 — Page-cache warmup curve.

Runs the same AirLLM generation N times (first = cold, subsequent = warm)
and records total runtime per run. Quantifies the OS page-cache speedup:
shards loaded on run 1 remain in the kernel's page cache → subsequent runs
skip the physical disk read for already-cached pages, revealing the
mmap/page-fault/page-cache mechanics from lecture Part-B.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from airllm_local_lab.sdk.model_loader.airllm_backend import AirLLMBackend
from airllm_local_lab.sdk.viz.plots import f6_page_cache_warmup
from airllm_local_lab.shared.config import load_config
from airllm_local_lab.shared.gatekeeper import Gatekeeper
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)
RESULTS = Path(__file__).resolve().parents[3] / "results"

TOTAL_RUNS = 5
COLD_RUNS = 1


def main() -> None:
    cfg = load_config()
    gk = Gatekeeper()
    RESULTS.mkdir(parents=True, exist_ok=True)

    model_id = cfg.model.sweep_model_id
    shards_path = str(Path(cfg.model.layer_shards_path).expanduser())
    prompt = cfg.benchmark.prompt_set[0]
    token = gk.hf_token()

    backend = AirLLMBackend(model_id=model_id, shards_path=shards_path, token=token)
    backend.load()

    runtimes: list[float] = []
    for i in range(1, TOTAL_RUNS + 1):
        t0 = time.perf_counter()
        backend.generate(prompt, max_new_tokens=cfg.model.max_new_tokens)
        elapsed = round(time.perf_counter() - t0, 3)
        runtimes.append(elapsed)
        cache_state = "cold" if i <= COLD_RUNS else "warm"
        log.info("E3 run %d/%d (%s): %.2f s", i, TOTAL_RUNS, cache_state, elapsed)

    backend.unload()

    cold = runtimes[:COLD_RUNS]
    warm = runtimes[COLD_RUNS:]
    speedup = cold[0] / warm[-1] if warm and warm[-1] > 0 else 1.0

    result = {"cold": cold, "warm": warm, "speedup": round(speedup, 2)}
    (RESULTS / "extension_e3.json").write_text(json.dumps(result, indent=2))
    f6_page_cache_warmup(cold, warm)

    log.info("E3 complete — speedup: %.2fx  results/extension_e3.json + assets/F6_page_cache_warmup.png", speedup)
