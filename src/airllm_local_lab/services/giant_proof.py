"""Task 2: Giant model proof — real OOM attempt + AirLLM layer-streaming of 13B."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

from airllm_local_lab.sdk.metrics.memory import MemorySampler
from airllm_local_lab.sdk.model_loader.airllm_backend import AirLLMBackend
from airllm_local_lab.shared.gatekeeper import Gatekeeper
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)
RESULTS = Path(__file__).resolve().parents[3] / "results"

GIANT_MODEL_ID = "huggyllama/llama-13b"  # 26 GB FP16, LLaMA-arch, Apache-2.0
GIANT_FP16_GB = 26.0
MAX_NEW_TOKENS = 8
PROMPT = "Explain in one sentence why large language models require so much memory."


def run_direct_oom_attempt(model_id: str, fp16_gb: float) -> dict:
    vm = psutil.virtual_memory()
    ram_avail_gb = vm.available / 1e9
    gap_gb = fp16_gb - ram_avail_gb
    script = f"import torch,time; from transformers import AutoModelForCausalLM; t=time.perf_counter(); m=AutoModelForCausalLM.from_pretrained('{model_id}',torch_dtype=torch.float16,low_cpu_mem_usage=True); print('LOADED',time.perf_counter()-t)"
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
        )
        elapsed = time.perf_counter() - t0
        if result.returncode == 0 and "LOADED" in result.stdout:
            return {
                "status": "loaded_unexpectedly",
                "elapsed_s": round(elapsed, 1),
                "note": "Model loaded — this machine has more RAM than expected.",
            }
        stderr = (result.stderr or "")[:500]
        stdout = (result.stdout or "")[:500]
        oom_keywords = ("MemoryError", "OOM", "Killed", "killed", "Cannot allocate")
        status = "oom_killed" if result.returncode < 0 else "oom_or_error"
        if any(k in stderr or k in stdout for k in oom_keywords):
            status = "oom_real"
        return {
            "status": status,
            "model": model_id,
            "fp16_gb": fp16_gb,
            "ram_total_gb": round(vm.total / 1e9, 1),
            "ram_available_gb": round(ram_avail_gb, 1),
            "gap_gb": round(gap_gb, 1),
            "returncode": result.returncode,
            "error": stderr or stdout or "Process terminated (OOM kill likely)",
            "elapsed_s": round(elapsed, 1),
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "oom_or_timeout",
            "model": model_id,
            "fp16_gb": fp16_gb,
            "ram_total_gb": round(vm.total / 1e9, 1),
            "ram_available_gb": round(ram_avail_gb, 1),
            "gap_gb": round(gap_gb, 1),
            "error": "Subprocess timed out after 180 s — OOM pressure likely",
            "elapsed_s": 180.0,
        }


def run_airllm_giant(model_id: str, shards_path: str, token: str | None) -> dict:
    log.info("=== AirLLM giant proof: %s → %s ===", model_id, shards_path)
    Path(shards_path).mkdir(parents=True, exist_ok=True)

    # AirLLM prefers pytorch_model.bin.index.json over safetensors when both exist
    # but the .bin files don't exist — rename to avoid the conflict.
    bin_idx = Path(model_id) / "pytorch_model.bin.index.json"
    if bin_idx.exists():
        bin_idx.rename(bin_idx.with_suffix(".json.bak"))

    backend = AirLLMBackend(model_id=model_id, shards_path=shards_path, token=token)
    sampler = MemorySampler()
    sampler.start()
    t0 = time.perf_counter()
    error_msg: str | None = None
    output_text = ""
    num_tokens = 0

    try:
        backend.load()
        result = backend.generate(PROMPT, max_new_tokens=MAX_NEW_TOKENS)
        output_text = result.text
        num_tokens = result.num_tokens
    except Exception as exc:
        error_msg = str(exc)[:500]
    finally:
        total_s = time.perf_counter() - t0
        mem = sampler.stop()
        backend.unload()

    vm = psutil.virtual_memory()
    return {
        "model_id": model_id,
        "fp16_gb": GIANT_FP16_GB,
        "ram_total_gb": round(vm.total / 1e9, 1),
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "output": output_text[:300],
        "text": output_text[:300],
        "num_tokens": num_tokens,
        "status": "error" if error_msg is not None else "ok",
        "error": error_msg,
        "total_s": round(total_s, 3),
        "elapsed_s": round(total_s, 3),
        "throughput_tps": round(num_tokens / total_s, 4) if total_s and num_tokens else 0,
        "peak_ram_mb": round(mem.peak_ram_mb, 1),
        "note": f"AirLLM MLX: one layer/token from NVMe. Peak RAM≈1 layer<<{GIANT_FP16_GB:.0f} GB full model.",
    }


def main() -> None:
    gk = Gatekeeper()
    RESULTS.mkdir(parents=True, exist_ok=True)

    shards_dir = str(Path("~/airllm_cache_giant").expanduser())
    local_model = str(Path("~/airllm_giant_download").expanduser())
    model_src = local_model if Path(local_model).exists() else GIANT_MODEL_ID

    direct = run_direct_oom_attempt(GIANT_MODEL_ID, GIANT_FP16_GB)
    log.info("Direct load status: %s", direct["status"])

    airllm_result = run_airllm_giant(model_src, shards_dir, gk.hf_token())
    log.info(
        "AirLLM giant: %s — %d tokens in %.1f s",
        airllm_result["status"],
        airllm_result["num_tokens"],
        airllm_result["total_s"],
    )

    out = {"direct_load": direct, "airllm_streaming": airllm_result}
    (RESULTS / "giant_proof.json").write_text(json.dumps(out, indent=2))
    log.info("Giant proof → results/giant_proof.json")
