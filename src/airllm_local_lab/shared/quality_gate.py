"""Quality gate: ruff + pytest-cov + 150-line check + basic secret scan."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
MAX_LINES = 150


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    result = subprocess.run(cmd, cwd=ROOT)
    ok = result.returncode == 0
    print(f"→ {'PASS' if ok else 'FAIL'}")
    return ok


def check_line_lengths() -> bool:
    violations: list[str] = []
    for py in SRC.rglob("*.py"):
        lines = py.read_text(encoding="utf-8").splitlines()
        if len(lines) > MAX_LINES:
            violations.append(f"  {py.relative_to(SRC.parent)}: {len(lines)} lines")
    if violations:
        print(f"\n{'=' * 60}\nLine-length check (max {MAX_LINES})\n{'=' * 60}")
        print("\n".join(violations))
        print("→ FAIL")
        return False
    print(f"\n{'=' * 60}\nLine-length check (max {MAX_LINES})\n{'=' * 60}\n→ PASS")
    return True


def secret_scan() -> bool:
    import re

    pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")
    hits: list[str] = []
    for py in SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if pattern.search(text):
            hits.append(str(py.relative_to(SRC.parent)))
    if hits:
        print(f"\n{'=' * 60}\nSecret scan\n{'=' * 60}")
        print("POTENTIAL SECRETS IN:", hits)
        print("→ FAIL")
        return False
    print(f"\n{'=' * 60}\nSecret scan\n{'=' * 60}\n→ PASS")
    return True


def main() -> None:
    results = [
        _run(["uv", "run", "ruff", "check", "."], "Ruff lint"),
        _run(["uv", "run", "ruff", "format", "--check", "."], "Ruff format"),
        check_line_lengths(),
        secret_scan(),
        _run(["uv", "run", "pytest", "--cov=airllm_local_lab", "--cov-fail-under=85", "-q"], "pytest + coverage"),
    ]
    if not all(results):
        print("\nQuality gate FAILED")
        sys.exit(1)
    print("\nQuality gate PASSED")
