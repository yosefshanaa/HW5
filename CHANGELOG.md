# Changelog

All notable changes to `airllm-local-lab` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.11] — 2026-06-20

### Added — F5 per-layer load-vs-compute timeline (real measured data)
- **`sdk/metrics/airllm_instrument.py`** — `capture_layer_timeline()` context manager monkey-patches AirLLM's MLX backend at two hot spots: `MlxModelPersister.load_model` (per-layer shard disk stream → `load_ms`) and `TransformerBlock.__call__` (per-layer forward → `compute_ms`). MLX's lazy graph is forced with `mx.eval` so each timed window captures the real disk read / matmuls. Only the first forward pass (prompt processing) is recorded — one clean sweep of every transformer layer. Also provides `write_timeline`/`read_timeline` (JSON persistence) and `io_fraction_of`.
- **`services/_report_figs.py`** — `render_f5()` regenerates `assets/F5_layer_timeline.png` from the measured JSON and returns an honest, data-derived caption + §7 lead-in (mean load/compute per layer, measured I/O fraction).
- New artifact: **`results/layer_timeline.json`** — captured live on Apple M3 Pro (TinyLlama-1.1B, 22 layers): mean **14.7 ms disk-load** vs **6.3 ms forward compute** per layer → **I/O fraction 70%** (disk-I/O-bound, not compute-bound). Honours ADR-008: figures from measured data only.

### Changed
- `shared/version.py` → 1.11
- `services/airllm_runner.py` — `run_airllm_demo()` now wraps generation in `capture_layer_timeline()` and writes `results/layer_timeline.json` (`uv run airllm-demo`).
- `services/benchmark_pipeline.py` — replaced the hardcoded `plots.f5_layer_timeline([])` with a read of `results/layer_timeline.json` (empty-list fallback retained when absent).
- `services/report_builder.py` — F5 figure regenerated + honest measured caption/blurb in §7 Benchmarking; footer date → 2026-06-20.
- `sdk/model_loader/airllm_backend.py` — removed the unused `_LAYER_EVENTS` placeholder (superseded by the real recorder).
- Docs updated: TODO Phase 10, this CHANGELOG entry.

---

## [1.10] — 2026-06-18

### Added — Honesty gap closures (three real experiments)
- **Task 1 (Ollama quant sweep):** `services/quant_sweep_ollama.py` — Measures ≥3 GGUF precision levels (Q8_0, Q4_K_M, Q2_K) of `llama3.2:1b` via Ollama HTTP API (`/api/generate stream=false`). Captures real TTFT, TPOT, throughput per precision with ≥3 reps + median/IQR. Honest framing: "AirLLM sub-FP16 quant is CUDA-only; precision axis measured via Ollama GGUF on Metal." Results → `results/quant_sweep_ollama.json/csv`.
- **Task 2 (Giant model proof):** `services/giant_proof.py` — (a) Real OOM attempt: `huggyllama/llama-13b` (26 GB FP16 > 18 GB total RAM) via HF Transformers subprocess → captured real OOM behavior. (b) AirLLM layer-streaming run of the same 13B model at `max_new_tokens=8` — proves layer-by-layer streaming works where direct load fails. Results → `results/giant_proof.json` + `results/baseline.json` updated.
- **Task 3 (Empirical TPOT/ITL):** `services/tpot_sweep.py` — Runs AirLLM at token counts 1,2,4,8; derives TPOT by linear fit on Δtime/Δtokens. Replaces TPOT=0 placeholder with measured ITL. Results → `results/tpot_sweep.json`. Ollama TPOT from Task 1 provides cross-backend reference.
- `sdk/model_loader/ollama_backend.py` — Added `generate_with_metrics()` (HTTP API with timing fields), `get_model_size_mb()` (from /api/ps), and `pull()` (ensures model available before sweep).
- New pyproject.toml entrypoints: `sweep-ollama`, `tpot-sweep`, `giant-proof`.

### Changed
- `shared/version.py` → 1.10
- Report builder + section builders updated with Ollama quant data, real OOM evidence, measured TPOT.
- KPI scorecard updated: K2 now cites ≥3 measured levels (Ollama GGUF); K3 cites measured TPOT from both Ollama and AirLLM ITL derivation; TPOT=0 caveat replaced.
- Docs updated: PLAN (ADR-004 revised, new ADR-009), PRD (FR-11/13/14, K2/K3, AC-8/9 resolved OQs), TODO (Phase 9 tasks), PRD_quantization.md, PRD_benchmarking.md, PRD_airllm.md, prompt_engineering_log.md.

---

## [1.00] — 2026-06-17

### Added
- Full project scaffold: `pyproject.toml`, `uv.lock`, `.python-version` (3.12), `.env-example`
- `shared/`: `gatekeeper`, `config`, `preflight`, `version`, `logging`, `quality_gate`
- `sdk/model_loader/`: `base` (Backend protocol), `airllm_backend`, `hf_backend`, `ollama_backend`
- `sdk/metrics/`: `timing` (TTFT/TPOT), `memory` (psutil sampler), `energy` (TDP × runtime), `layer_timeline`
- `sdk/quality/rater`: deterministic rubric scorer (coherence / correctness / completeness)
- `sdk/economics/`: `onprem`, `api` (with Prompt Caching), `breakeven`
- `sdk/viz/`: `plots` (F1–F7, break-even chart), `tables` (Markdown tables)
- `services/`: `baseline_runner`, `airllm_runner`, `quant_sweep`, `benchmark_pipeline`, `economic_model`, `report_builder`
- `services/extension_e1_io`: shard-location I/O sensitivity (macOS SSD vs /tmp)
- `services/extension_e3_pagecache`: cold→warm page-cache warmup curve
- `scripts/download`: dry-run + free-space guard for HF model downloads
- `config/`: `default.toml`, `models.toml`, `economics.toml`
- Unit tests: timing, memory, energy, layer_timeline, quality/rater, economics (onprem/api/breakeven), gatekeeper, config, viz/tables
- `docs/PRD.md`, `PLAN.md`, `TODO.md`, `PRD_airllm.md`, `PRD_quantization.md`, `PRD_benchmarking.md`, `PRD_economic_analysis.md`
- macOS/Apple Silicon adaptation: MPS detection, native NVMe I/O, no WSL2 dependencies
