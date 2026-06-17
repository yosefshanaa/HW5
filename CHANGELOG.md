# Changelog

All notable changes to `airllm-local-lab` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
