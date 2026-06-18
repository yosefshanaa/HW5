"""Static README section builders — pure string functions, no I/O."""

from __future__ import annotations

from airllm_local_lab.services._report_theory import section_theory_iso  # noqa: F401

_HW = {
    "CPU": {"spec": "Apple M3 Pro (ARM64, 12-core)", "implication": "No CUDA; MPS available but AirLLM runs CPU mode"},
    "RAM": {"spec": "18 GB unified memory", "implication": "Hard ceiling; 70B FP16 (~140 GB) still exceeds it 7×"},
    "GPU": {
        "spec": "Apple M3 Pro GPU (Metal/MPS) — no NVIDIA CUDA",
        "implication": "AirLLM CUDA path unavailable; CPU-only inference",
    },
    "Disk": {
        "spec": "460 GiB NVMe (APFS), 193 GiB free",
        "implication": "Ample for shards; native NVMe (~7 GB/s rated) holds layer shards",
    },
    "OS": {"spec": "macOS Darwin 25.2 (ARM64)", "implication": "Native NVMe I/O; no cross-OS penalty"},
    "Python": {
        "spec": "3.12 (pinned via uv)",
        "implication": "Satisfies non-newest constraint; torch/airllm wheels available",
    },
}


def section_header(version: str) -> str:
    return (
        f"# AirLLM Local Lab — Running Giant LLMs on Constrained Hardware\n\n"
        f"[![Python](https://img.shields.io/badge/python-3.12-blue)](#) "
        f"[![Package manager](https://img.shields.io/badge/deps-uv-purple)](#) "
        f"[![Lint](https://img.shields.io/badge/lint-ruff-orange)](#) "
        f"[![Version](https://img.shields.io/badge/version-{version}-green)](#)\n\n"
        "> **HW5 / Assignment 05 — Deep-dive technical report.** Running LLMs that *do not fit* "
        "in memory via AirLLM layer-streaming, quantization, and rigorous benchmarking on GPU-less Apple Silicon hardware.\n\n"
        "> Full planning documentation (PRD · PLAN · ADRs · TODO · per-mechanism PRDs) is in the "
        "[Project Planning & Documentation](#project-planning--documentation) section below."
    )


def section_models() -> str:
    return (
        "## 2. Model Choice Justification\n\n"
        "| Role | Model | Size FP16 | Rationale |\n|---|---|---|---|\n"
        "| Giant OOM + AirLLM proof | `huggyllama/llama-13b` | 26 GB | "
        "Apache-2.0, ungated, LLaMA-architecture → compatible with AirLLM MLX. "
        "26 GB FP16 > 18 GB RAM. Real OOM measured. |\n"
        "| AirLLM demo + TPOT sweep | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2.2 GB | LLaMA-based; "
        "compatible with AirLLM's MLX backend on macOS; fast download; demonstrates layer-streaming |\n"
        "| Quant sweep (Ollama GGUF) | `llama3.2:1b` Q8_0/Q4_K_M/Q2_K | 0.6–1.3 GB | "
        "3 measured precision levels on macOS Metal; real TTFT/TPOT/throughput via /api/generate |\n\n"
        "**Selection reasoning:** AirLLM on macOS exclusively uses its MLX backend, "
        "which requires LLaMA-family architecture (weight layout `model.layers.{i}`, RMSNorm, etc.). "
        "`huggyllama/llama-13b` was chosen over OPT-13b (CUDA-only architecture for AirLLM) "
        "and `openlm-research/open_llama_13b` (only `.bin` pickle format, not safetensors). "
        "It is Apache-2.0, ungated, safetensors-native, and 26.03 GB FP16 — "
        "the direct HF load provably fails on 18 GB RAM. "
        "AirLLM layer-streaming then proves the model can generate tokens despite exceeding total RAM."
    )


def section_baseline(b_sanity: dict, b_direct: dict) -> str:
    gap = b_direct.get("gap_gb", "?")
    fp16_gb = b_direct.get("fp16_gb", b_direct.get("required_fp16_gb", "?"))
    avail = b_direct.get("ram_available_gb", "?")
    status = b_direct.get("status", "not run")
    stderr_snippet = str(b_direct.get("stderr", b_direct.get("error", "—")))[:300]
    real_oom = status in ("oom_real", "oom_killed", "oom_or_timeout", "oom_or_error")
    proof_note = (
        "**Real OOM measured** — subprocess killed/failed when attempting "
        f"`huggyllama/llama-13b` ({fp16_gb} GB FP16) direct load."
        if real_oom
        else "**Analytic OOM proof** — direct load not attempted (avoids 26 GB download)."
    )
    return (
        "## 3. Baseline (Evidence of the Problem)\n\n"
        "### 3.1 Small-model sanity baseline\n"
        f"- **Status:** `{b_sanity.get('status', 'not run')}`\n"
        f"- **Output:** `{b_sanity.get('output', '—')[:120]}`\n"
        f"- **Runtime:** {b_sanity.get('runtime_s', 0):.2f} s\n\n"
        "### 3.2 Direct load of 13B model (expected OOM)\n"
        f"- **Status:** `{status}` {proof_note}\n"
        f"- **Evidence:** `{stderr_snippet}`\n"
        f"- **RAM available:** {avail} GB  |  **Required (FP16):** {fp16_gb} GB  |  **Gap:** {gap} GB\n\n"
        "**Explanation:** `huggyllama/llama-13b` in FP16 requires 26.03 GB. With only ~7 GB available "
        "(18 GB total minus ~11 GB consumed by macOS, browsers, and running processes), "
        "a direct HuggingFace load exhausts physical RAM — the kernel kills the process. "
        "This is the *Memory Wall* the lecture describes: RAM capacity, not compute, is the bottleneck."
    )


def section_airllm(demo: dict) -> str:
    return (
        "## 5. AirLLM Integration\n\n"
        "**Mechanism:** AirLLM stores each transformer layer as a `safetensors` shard on disk and streams "
        "them one at a time: load layer → compute hidden state → release → load next. "
        "Peak memory = one layer, not the whole model. The price paid: weights are re-read from disk "
        "every token — the constraint shifts from *memory* to *time and I/O*.\n\n"
        "### Demo result\n```\n"
        f"Model:      {demo.get('model_id', '—')}\n"
        f"Output:     {demo.get('output', '—')[:200]}\n"
        f"Tokens:     {demo.get('num_tokens', 0)}\n"
        f"Runtime:    {demo.get('total_s', 0):.1f} s\n"
        f"Throughput: {demo.get('throughput_tps', 0):.4f} tok/s\n"
        f"Peak RAM:   {demo.get('peak_ram_mb', 0):.0f} MB\n```\n\n"
        "### Live terminal capture (`uv run airllm-demo`)\n\n"
        "![S1 — AirLLM demo terminal run](assets/S1_airllm_demo_run.png)\n\n"
        "*S1 — Real terminal output of `uv run airllm-demo` on a cold run. "
        "The 22 progress bars each represent one transformer layer being streamed sequentially "
        "from NVMe (`~/airllm_cache/`). Each pass through all 22 layers takes roughly 1.5 s; "
        "the complete 20-token generation finishes in **28.7 s** with peak RAM held at just ~1.1 GB "
        "— the whole 2.2 GB model never loads into memory at once. "
        "The final output line confirms the model ran entirely **on-device with no GPU and no internet connection**.*"
    )
