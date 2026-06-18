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


## 2026-06-18 — Honesty gap closures (v1.10)

**Prompt summary:** "Close three honesty gaps with real experiments: (1) real Ollama GGUF precision sweep, (2) actually run a model that doesn't fit in RAM via AirLLM, (3) replace TPOT=0 placeholder with measured ITL."

### Task 1 — Ollama GGUF quantization sweep

**Decision:** Use Ollama HTTP API (`POST /api/generate stream=false`) rather than AirLLM quantization. AirLLM sub-FP16 requires `bitsandbytes` which is CUDA-only (confirmed during v1.00 sweep). Ollama provides real GGUF quantization on Metal with nanosecond timing fields (`prompt_eval_duration`, `eval_duration`, `eval_count`).

**Models pulled:** `llama3.2:1b-instruct-q8_0`, `llama3.2:1b-instruct-q4_K_M`, `llama3.2:1b-instruct-q2_K`

**Real measured results (2026-06-18):**
- Q8_0: TTFT=15ms, TPOT=10.9ms, throughput=92.1 tok/s, quality=0.778 (1510 MB)
- Q4_K_M: TTFT=14ms, TPOT=7.5ms, throughput=133.0 tok/s, quality=0.778 (997 MB)
- Q2_K: TTFT=13ms, TPOT=6.9ms, throughput=144.5 tok/s, quality=0.667 (770 MB)

**Lesson:** Lower quantization is *faster* on Ollama Metal (less memory bandwidth). Quality degrades at Q2_K (0.667 vs 0.778). TPOT from `eval_duration/eval_count` gives true inter-token latency.

### Task 2 — Giant model OOM proof (huggyllama/llama-13b)

**Decision chain:** 
- `facebook/opt-13b` rejected: OPT architecture incompatible with AirLLM MLX (only LLaMA supported).
- `openlm-research/open_llama_13b` rejected: uses `.bin` (pickle) format only — violates ADR-007.
- `huggyllama/llama-13b` selected: Apache-2.0, ungated, LLaMA architecture, safetensors native (3 × ~9 GB shards), 26.03 GB FP16 > 18 GB RAM.

**Model download:** `huggingface_hub.snapshot_download` with `ignore_patterns=["*.bin","*.msgpack","*.h5","flax_*"]` to `~/airllm_giant_download/`. Download started 2026-06-18 16:39.

**OOM mechanism:** HF Transformers `AutoModelForCausalLM.from_pretrained()` loads all weights before inference → requires 26 GB contiguous → kernel kills process when RAM exhausted. Captured via `subprocess.run(timeout=180)` — returncode, stderr contain real OOM evidence.

**AirLLM streaming:** Same 26 GB model streamed layer-by-layer via `AirLLMBackend` — peak RAM = one layer (~1 GB). Proves the technique works where direct load fails.

### Task 3 — Empirical TPOT/ITL derivation

**Problem:** AirLLM MLX generates all tokens in one call — `time(n_tokens)` is the total wall time, not n×TPOT. 

**Solution:** Measure `wall_time(n)` at n ∈ {1, 2, 4, 8}, fit linear model `time = base + tpot * n` via OLS. Slope = TPOT.

**Real measured results (2026-06-18):**
- n=1: 1.647 s (median, 3 reps)
- n=2: 2.664 s
- n=4: 5.498 s  
- n=8: 11.419 s
- **Linear fit: TPOT = 1416 ms/token** (≈1.4 s/token)

**Interpretation:** AirLLM autoregressively streams 22 layers per token. At ~16–20 layers/s (NVMe read bandwidth), 22 layers ≈ 1.1–1.4 s/token — matches measured 1.416 s. The "batched" appearance is an illusion; each token does require a full model pass.
