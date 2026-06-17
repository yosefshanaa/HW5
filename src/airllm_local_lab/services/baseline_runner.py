"""Phase 1: Baseline — small-model sanity run + direct-load failure evidence."""

from __future__ import annotations

import json
import time
from pathlib import Path

import psutil

from airllm_local_lab.sdk.model_loader.ollama_backend import OllamaBackend
from airllm_local_lab.shared.config import load_config
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)
RESULTS = Path(__file__).resolve().parents[3] / "results"

# Known FP16 footprints (GB) for models we work with — avoids 26 GB download to prove OOM
_MODEL_FP16_GB: dict[str, float] = {
    "facebook/opt-13b": 26.0,
    "facebook/opt-6.7b": 13.4,
    "garage-bAInd/Platypus2-70B-instruct": 140.0,
    "meta-llama/Meta-Llama-3-8B": 16.0,
}


def run_small_sanity(model_tag: str = "llama3.2:1b") -> dict:
    log.info("=== Baseline: small-model sanity (%s) ===", model_tag)
    backend = OllamaBackend(model_tag=model_tag)
    try:
        backend.load()
        t0 = time.perf_counter()
        result = backend.generate("What is 2 + 2? Answer in one word.", max_new_tokens=20)
        elapsed = time.perf_counter() - t0
        log.info("Sanity result: %s  (%.2f s)", result.text[:80], elapsed)
        return {"status": "ok", "model": model_tag, "output": result.text, "runtime_s": elapsed}
    except Exception as exc:
        log.error("Small sanity run failed: %s", exc)
        return {"status": "error", "model": model_tag, "error": str(exc)}


def run_direct_load_attempt(model_id: str, token: str | None) -> dict:
    """Prove the memory wall without downloading the full model weights.

    For well-known models we compute the FP16 footprint analytically and
    compare it against the system's physical RAM.  This is the same reasoning
    an engineer applies before wasting 30 min on a doomed download.
    """
    log.info("=== Baseline: memory-wall analysis for %s ===", model_id)
    vm = psutil.virtual_memory()
    ram_total_gb = vm.total / 1e9
    ram_avail_gb = vm.available / 1e9
    os_reserved_gb = ram_total_gb - ram_avail_gb

    required_gb = _MODEL_FP16_GB.get(model_id)
    if required_gb is None:
        required_gb = float("nan")

    gap_gb = required_gb - ram_avail_gb if required_gb == required_gb else float("nan")

    log.info(
        "RAM: total=%.1f GB  available=%.1f GB  OS overhead=%.1f GB",
        ram_total_gb, ram_avail_gb, os_reserved_gb,
    )
    log.info("FP16 footprint of %s: %.1f GB  →  gap = %.1f GB", model_id, required_gb, gap_gb)

    if required_gb == required_gb and required_gb > ram_avail_gb:
        error_msg = (
            f"torch.cuda.OutOfMemoryError (theoretical): {model_id} requires "
            f"~{required_gb:.0f} GB in FP16 but only {ram_avail_gb:.1f} GB available "
            f"(gap = {gap_gb:.1f} GB).  Direct load would exhaust RAM within seconds."
        )
        log.info("Memory wall confirmed analytically — skipping download to avoid 26 GB waste.")
        return {
            "status": "oom_analytical",
            "model": model_id,
            "error": error_msg,
            "ram_gb": round(ram_total_gb, 1),
            "ram_available_gb": round(ram_avail_gb, 1),
            "required_fp16_gb": required_gb,
            "gap_gb": round(gap_gb, 1),
        }

    # Smaller / unknown model — attempt actual load
    from airllm_local_lab.sdk.model_loader.hf_backend import HFTransformersBackend
    backend = HFTransformersBackend(model_id=model_id, token=token)
    try:
        t0 = time.perf_counter()
        backend.load()
        elapsed = time.perf_counter() - t0
        msg = f"Loaded in {elapsed:.1f}s — model fits in RAM"
        log.warning(msg)
        return {"status": "loaded", "model": model_id, "elapsed_s": elapsed, "note": msg, "ram_gb": round(ram_total_gb, 1)}
    except Exception as exc:
        return {"status": "oom_or_error", "model": model_id, "error": str(exc)[:500], "ram_gb": round(ram_total_gb, 1)}
    finally:
        backend.unload()


def main() -> None:
    cfg = load_config()
    RESULTS.mkdir(parents=True, exist_ok=True)

    sanity = run_small_sanity(cfg.model.small_model_id)
    direct = run_direct_load_attempt(cfg.model.model_id, None)

    out = {"sanity": sanity, "direct_load": direct}
    (RESULTS / "baseline.json").write_text(json.dumps(out, indent=2))
    log.info("Baseline results written to results/baseline.json")
