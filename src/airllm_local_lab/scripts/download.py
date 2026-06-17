"""Download helper with dry-run, file-size preview, and free-space guard."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from airllm_local_lab.shared.gatekeeper import Gatekeeper
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)


def _free_gb(path: str) -> float:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(p).free / 1e9


def list_model_files(model_id: str, token: str | None) -> list[dict]:
    from huggingface_hub import list_repo_files

    files = []
    for fname in list_repo_files(model_id, token=token):
        files.append({"name": fname})
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="AirLLM model download helper")
    parser.add_argument("model_id", nargs="?", default="garage-bAInd/Platypus2-70B-instruct")
    parser.add_argument("--cache-dir", default="~/airllm_cache")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    args = parser.parse_args()

    gk = Gatekeeper()
    token = gk.hf_token()
    free = _free_gb(args.cache_dir)
    log.info("Free disk at %s: %.1f GB", args.cache_dir, free)

    if free < args.min_free_gb:
        log.error("Only %.1f GB free — need ≥ %.0f GB. Aborting.", free, args.min_free_gb)
        sys.exit(1)

    if args.dry_run:
        log.info("DRY RUN — listing files for %s", args.model_id)
        try:
            files = list_model_files(args.model_id, token)
            for f in files[:30]:
                print(f"  {f['name']}")
            if len(files) > 30:
                print(f"  ... and {len(files) - 30} more")
        except Exception as exc:
            log.warning("Could not list files: %s", exc)
        return

    log.info("Starting download of %s → %s", args.model_id, args.cache_dir)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(Path(args.cache_dir).expanduser() / args.model_id.replace("/", "--")),
        token=token,
        ignore_patterns=["*.bin", "original/"],
    )
    log.info("Download complete.")
