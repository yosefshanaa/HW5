"""Phase 1: Baseline — small-model sanity run + direct-load failure evidence."""

from __future__ import annotations

import json
import time
from pathlib import Path

from airllm_local_lab.sdk.model_loader.hf_backend import HFTransformersBackend
from airllm_local_lab.sdk.model_loader.ollama_backend import OllamaBackend
from airllm_local_lab.shared.config import load_config
from airllm_local_lab.shared.gatekeeper import Gatekeeper
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)
RESULTS = Path(__file__).resolve().parents[4] / "results"


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
    log.info("=== Baseline: direct-load attempt for %s ===", model_id)
    import psutil

    ram_gb = psutil.virtual_memory().total / 1e9
    log.info("Available RAM: %.1f GB", ram_gb)

    backend = HFTransformersBackend(model_id=model_id, token=token)
    try:
        t0 = time.perf_counter()
        backend.load()
        elapsed = time.perf_counter() - t0
        msg = f"UNEXPECTED SUCCESS in {elapsed:.1f}s — model may be smaller than expected"
        log.warning(msg)
        return {"status": "loaded", "model": model_id, "elapsed_s": elapsed, "note": msg}
    except Exception as exc:
        evidence = str(exc)
        log.info("Expected failure: %s", evidence[:200])
        return {
            "status": "oom_or_error",
            "model": model_id,
            "error": evidence[:500],
            "ram_gb": round(ram_gb, 1),
        }
    finally:
        backend.unload()


def main() -> None:
    cfg = load_config()
    gk = Gatekeeper()
    RESULTS.mkdir(parents=True, exist_ok=True)

    sanity = run_small_sanity(cfg.model.small_model_id)
    direct = run_direct_load_attempt(cfg.model.model_id, gk.hf_token())

    out = {"sanity": sanity, "direct_load": direct}
    (RESULTS / "baseline.json").write_text(json.dumps(out, indent=2))
    log.info("Baseline results written to results/baseline.json")
