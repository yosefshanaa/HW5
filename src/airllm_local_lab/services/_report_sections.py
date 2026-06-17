"""Static README section builders — pure string functions, no I/O."""

from __future__ import annotations

_HW = {
    "CPU": {"spec": "Apple M3 Pro (ARM64, 12-core)", "implication": "No CUDA; MPS available but AirLLM runs CPU mode"},
    "RAM": {"spec": "18 GB unified memory", "implication": "Hard ceiling; 70B FP16 (~140 GB) still exceeds it 7×"},
    "GPU": {
        "spec": "Apple M3 Pro GPU (Metal/MPS) — no NVIDIA CUDA",
        "implication": "AirLLM CUDA path unavailable; CPU-only inference",
    },
    "Disk": {
        "spec": "460 GiB NVMe (APFS), 193 GiB free",
        "implication": "Ample for shards; native NVMe faster than WSL2 9p",
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
        "## Planning documents\n"
        "| Doc | Purpose |\n|---|---|\n"
        "| [docs/PRD.md](docs/PRD.md) | Requirements, KPIs, acceptance criteria |\n"
        "| [docs/PLAN.md](docs/PLAN.md) | Architecture, C4, ADRs |\n"
        "| [docs/TODO.md](docs/TODO.md) | Phased backlog |"
    )


def section_models() -> str:
    return (
        "## 2. Model Choice Justification\n\n"
        "| Role | Model | Size FP16 | Rationale |\n|---|---|---|---|\n"
        "| Giant (proof) | `garage-bAInd/Platypus2-70B-instruct` | ~140 GB | Demonstrably doesn't fit; K1 target |\n"
        "| Sweep | `meta-llama/Meta-Llama-3-8B` | ~16 GB | Exceeds 18 GB RAM w/ OS overhead; tractable on CPU |\n"
        "| Sanity baseline | `llama3.2:1b` (Ollama) | <2 GB | Validates harness; fits trivially |"
    )


def section_baseline(b_sanity: dict, b_direct: dict) -> str:
    return (
        "## 3. Baseline (Evidence of the Problem)\n\n"
        "### 3.1 Small-model sanity baseline\n"
        f"- **Status:** `{b_sanity.get('status', 'not run')}`\n"
        f"- **Output:** `{b_sanity.get('output', '—')[:120]}`\n"
        f"- **Runtime:** {b_sanity.get('runtime_s', 0):.2f} s\n\n"
        "### 3.2 Direct load of 70B model (expected failure)\n"
        f"- **Status:** `{b_direct.get('status', 'not run')}`\n"
        f"- **Error:** `{str(b_direct.get('error', '—'))[:300]}`\n"
        f"- **RAM available:** {b_direct.get('ram_gb', 0):.1f} GB\n"
        "- **Explanation:** A 70B model in FP16 requires ~140 GB of RAM. With 18 GB available "
        "(and OS/process overhead reducing usable headroom), the direct load exhausts memory immediately "
        "— the RAM gap is ~122 GB. This is the *Memory Wall* the lecture describes."
    )


def section_airllm(demo: dict) -> str:
    return (
        "## 4. AirLLM Integration\n\n"
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
        f"Peak RAM:   {demo.get('peak_ram_mb', 0):.0f} MB\n```"
    )


def section_theory_iso(f7_md: str, f6_md: str) -> str:
    return (
        "## 8. Theory Linkage\n\n"
        "| Empirical finding | Theoretical mechanism |\n|---|---|\n"
        "| TTFT is disproportionately long | Prefill is compute-bound (GEMM); builds full KV cache before first token |\n"
        "| TPOT dominated by I/O, not FLOPs | Decode is memory-bandwidth-bound (GEMV) → AirLLM makes it disk-I/O-bound |\n"
        "| Warm runs faster than cold | OS page cache retains recently used shard pages (mmap + page-fault mechanics) |\n"
        "| Lower precision → lower RAM + faster I/O | Fewer bits per weight → smaller shard → less I/O per layer |\n"
        "| Single-layer peak memory | AirLLM's layer-streaming trades the memory limit for a time limit |\n\n"
        "## 9. Extension E1 — Shard-location I/O Sensitivity\n\n"
        "AirLLM's bottleneck is disk I/O. This extension benchmarks identical runs with shards on:\n"
        "- **Internal NVMe SSD** (~/airllm_cache) — native macOS APFS\n"
        "- **RAM disk** (/tmp fast path) — for upper-bound comparison\n\n"
        f"{f7_md}\n"
        "## 10. Extension E3 — Page-Cache Warmup Curve\n\n"
        "The OS page cache retains recently loaded shard pages. Extension E3 quantifies the cold→warm speedup.\n\n"
        f"{f6_md}\n\n"
        "## 11. ISO/IEC 25010 Mapping\n\n"
        "| Characteristic | How addressed |\n|---|---|\n"
        "| Functional suitability | Giant model generates coherent output (K1); all 8 metric families captured (K3) |\n"
        "| Performance efficiency | Benchmarked TTFT/TPOT/throughput; quantized to reduce per-layer I/O |\n"
        "| Reliability | ≥3 repetitions; median+IQR; cold/warm cache distinguished |\n"
        "| Security | Gatekeeper; HF token env-only; `.env` git-ignored; safetensors only |\n"
        "| Maintainability | TDD ≥85%; Ruff zero violations; ≤150 lines/file; SDK layering |\n"
        "| Portability | Device-agnostic (CPU/MPS/CUDA detection); uv locked env |"
    )
