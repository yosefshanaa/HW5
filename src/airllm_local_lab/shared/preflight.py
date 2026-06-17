"""Preflight checks: Python version, torch, device, disk, tokenizer."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from airllm_local_lab.shared.logging import get_logger
from airllm_local_lab.shared.version import __version__

log = get_logger(__name__)

MIN_DISK_GB = 50.0


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    ok = (major == 3) and (10 <= minor <= 12)
    status = "OK" if ok else f"WARN — Python {major}.{minor} (expected 3.10–3.12)"
    log.info("Python %s.%s — %s", major, minor, status)
    return ok


def check_torch() -> tuple[bool, str]:
    try:
        import torch

        cuda = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if mps:
            device = "mps"
        elif cuda:
            device = "cuda"
        else:
            device = "cpu"
        log.info("torch %s — device=%s  cuda=%s  mps=%s", torch.__version__, device, cuda, mps)
        return True, device
    except ImportError:
        log.error("torch not importable — run `uv sync`")
        return False, "unavailable"


def check_airllm() -> bool:
    try:
        from airllm import AutoModel  # noqa: F401

        log.info("AirLLM OK")
        return True
    except ImportError:
        log.warning("airllm not importable — will be needed for Phase 2")
        return False


def check_disk(path: str = "~") -> tuple[bool, float]:
    resolved = Path(path).expanduser()
    resolved.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(resolved)
    free_gb = usage.free / 1e9
    ok = free_gb >= MIN_DISK_GB
    log.info("Disk at %s — %.1f GB free  (need ≥ %.0f GB) — %s", resolved, free_gb, MIN_DISK_GB, "OK" if ok else "LOW")
    return ok, free_gb


def check_tokenizer(model_id: str, token: str | None = None) -> bool:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id, token=token, local_files_only=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        log.info("Tokenizer for %s — OK (pad_token set)", model_id)
        return True
    except Exception as exc:
        log.warning("Tokenizer check skipped for %s — %s", model_id, exc)
        return False


def run_all(shards_path: str = "~/airllm_cache") -> dict[str, object]:
    log.info("=== Preflight v%s ===", __version__)
    py_ok = check_python()
    torch_ok, device = check_torch()
    airllm_ok = check_airllm()
    disk_ok, free_gb = check_disk(shards_path)
    summary = {
        "python_ok": py_ok,
        "torch_ok": torch_ok,
        "device": device,
        "airllm_ok": airllm_ok,
        "disk_ok": disk_ok,
        "free_gb": free_gb,
    }
    all_ok = py_ok and torch_ok and disk_ok
    log.info("Preflight %s — device=%s  free_gb=%.1f", "PASS" if all_ok else "WARN", device, free_gb)
    return summary


def main() -> None:
    from airllm_local_lab.shared.config import load_config

    cfg = load_config()
    result = run_all(cfg.model.layer_shards_path)
    if not result["disk_ok"]:
        sys.exit(1)
