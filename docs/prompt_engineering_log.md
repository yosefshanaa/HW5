# Prompt Engineering / Vibe-Coding Log

Records significant prompts, model choices, and decisions made during development.

---

## 2026-06-17 — Initial scaffolding

**Prompt to Claude Code:** "Read the PRD files, TODO and PLAN carefully and report if I can do the project on my computer."

**Finding:** Hardware is Apple M3 Pro with 18 GB RAM and 193 GB free disk. No NVIDIA CUDA. AirLLM CPU/MPS mode is the path. Project is fully feasible — hardware is *more capable* than the PRD's WSL2 baseline, requiring adaptation of the WSL2-specific narrative.

**Decision recorded:** ADR-001 adapted — no CUDA, CPU mode + potential MPS fallback. Extension E1 adapted for macOS (internal NVMe vs /tmp).

---

## 2026-06-17 — Full implementation

**Prompt:** "Start implementing everything based on this chat message and the PRDs, plan, todo."

**Approach:** Implemented all 7 phases in order:
- P0: repo structure, uv, pyproject.toml, shared layer, gatekeeper, config, preflight, quality gate
- P1: baseline_runner (Ollama sanity + HF direct-load failure capture)
- P2: airllm_runner with AirLLM AutoModel on CPU, layer shards to ~/airllm_cache
- P3: quant_sweep (fp16 / 8bit / 4bit via AirLLM compression)
- P4: benchmark_pipeline with ≥3 reps, cold/warm, all 8 metric families, viz
- P5: economic_model (OnPrem CAPEX/OPEX vs Managed API with Prompt Caching, break-even chart)
- P6: extension_e1_io (I/O location), extension_e3_pagecache (warmup curve)
- P7: report_builder assembles README

**Key macOS adaptation decisions:**
- `LAYER_SHARDS_SAVING_PATH = ~/airllm_cache` (not /mnt/c as in PRD)
- Device detection: check `torch.backends.mps.is_available()` first, fall back to CPU
- Extension E1: compare internal SSD (~/) vs /tmp (potential RAM disk on macOS)
- Python 3.12 pinned via uv (system has 3.9 — uv downloads 3.12.13 aarch64)
- torch==2.2.2 for ARM64 compatibility with airllm

**Model choices (ADR-002):**
- Giant proof model: `garage-bAInd/Platypus2-70B-instruct` (~140 GB FP16, single "it runs" demo)
- Sweep model: `meta-llama/Meta-Llama-3-8B` (~16 GB FP16, full precision sweep + reps)
- Sanity baseline: `llama3.2:1b` via Ollama

---
