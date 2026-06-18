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
        "| OOM proof | `facebook/opt-13b` | ~26 GB | 26 GB > 18 GB RAM → direct load fails analytically |\n"
        "| AirLLM demo + sweep | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2.2 GB | LLaMA-based; "
        "compatible with AirLLM's MLX backend on macOS; fast download; demonstrates layer-streaming |\n"
        "| Sanity baseline | `llama3.2:1b` (Ollama) | <2 GB | Validates harness; fits trivially |\n\n"
        "**Selection reasoning:** AirLLM on macOS exclusively uses its MLX backend, "
        "which requires LLaMA-family architecture (weight layout `model.layers.{i}`, RMSNorm, etc.). "
        "OPT-family models use a different weight structure (`model.decoder.layers.{i}`, LayerNorm) "
        "and are incompatible with the MLX path. TinyLlama is a public LLaMA-1.1B model that: "
        "downloads quickly (~2.2 GB), exercises the full AirLLM layer-streaming pipeline "
        "(split → stream-load → generate → unload at transformer-layer granularity), and allows "
        "quantitative comparison of FP16/8bit/4bit precision trade-offs. "
        "OPT-13b retains its role as the analytic OOM proof: 26 GB FP16 vs. 18 GB RAM — "
        "direct load fails without downloading."
    )


def section_baseline(b_sanity: dict, b_direct: dict) -> str:
    gap = b_direct.get("gap_gb", "?")
    required = b_direct.get("required_fp16_gb", "?")
    avail = b_direct.get("ram_available_gb", "?")
    return (
        "## 3. Baseline (Evidence of the Problem)\n\n"
        "### 3.1 Small-model sanity baseline\n"
        f"- **Status:** `{b_sanity.get('status', 'not run')}`\n"
        f"- **Output:** `{b_sanity.get('output', '—')[:120]}`\n"
        f"- **Runtime:** {b_sanity.get('runtime_s', 0):.2f} s\n\n"
        "### 3.2 Direct load of OPT-13B (expected failure)\n"
        f"- **Status:** `{b_direct.get('status', 'not run')}`\n"
        f"- **Error:** `{str(b_direct.get('error', '—'))[:300]}`\n"
        f"- **RAM available:** {avail} GB  |  **Required (FP16):** {required} GB  |  **Gap:** {gap} GB\n\n"
        "**Explanation:** `facebook/opt-13b` in FP16 requires ~26 GB. With only ~7 GB available "
        "(18 GB total minus ~11 GB consumed by macOS, browsers, and running processes), "
        "a direct HuggingFace load would exhaust physical RAM within seconds — causing the kernel "
        "to page aggressively, thrashing the NVMe, and making inference infeasibly slow. "
        "This is the *Memory Wall* the lecture describes: RAM capacity, not compute, is the bottleneck. "
        "**Analytic proof avoids the 26 GB download** — same conclusion, zero waste."
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
