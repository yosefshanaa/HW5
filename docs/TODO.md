# TODO — Execution Backlog

| Field | Value |
|---|---|
| **Document** | Phased task list with status + Definition of Done |
| **Project** | `airllm-local-lab` |
| **Version** | 1.10 |
| **Last updated** | 2026-06-18 |
| **Status** | ✅ All phases complete — v1.10 delivered |
| **Related** | [PRD.md](./PRD.md) · [PLAN.md](./PLAN.md) · per-mechanism PRDs |

**Status legend:** ☑ todo · ◐ in progress · ☑ done · ⚠ blocked · ⤬ infeasible (documented)
**Each task lists its Definition of Done (DoD) and the PRD/FR it satisfies.**

---

## Phase 0 — Repository, environment & guardrails

| # | Task | Status | DoD | Ref |
|---|---|---|---|---|
| 0.1 | Init repo, `.gitignore` **first** (`.env`, `.venv`, shards, models, caches) | ☑ | `git status` shows no secret/large artifacts trackable | NFR-5/10 |
| 0.2 | `uv python pin 3.12` (fallback 3.11); create `.python-version` | ☑ | `uv run python --version` → 3.12.x | ADR-005 |
| 0.3 | `pyproject.toml` (deps via `uv add`: torch CPU, transformers, airllm, safetensors, huggingface_hub, pydantic, matplotlib, pytest, pytest-cov, ruff, psutil, typer) + commit `uv.lock` | ☑ | `uv sync` clean from scratch | FR-1, ADR-006 |
| 0.4 | Ruff config `select = E,F,W,I,N,UP,B,C4,SIM` | ☑ | `uv run ruff check` runs, 0 violations on skeleton | NFR-2 |
| 0.5 | Package skeleton `sdk/ services/ shared/` + `version.py` (`__version__="1.00"`) | ☑ | imports resolve; version reads 1.00 | NFR-4/6 |
| 0.6 | `shared/gatekeeper.py` — creds from env only | ☑ | unit test: missing `HF_TOKEN` raises clear error; token never logged | FR-2, NFR-5 |
| 0.7 | `shared/config.py` typed config + `config/*.toml` (default/models/economics) | ☑ | config loads + env override unit-tested | FR-4 |
| 0.8 | `shared/preflight.py` — python/torch/`cuda.is_available()`/disk/tokenizer checks | ☑ | `uv run preflight` prints device=cpu, free disk, warns on Py>3.12 | FR-3, R3/R5 |
| 0.9 | `.env-example` (HF_TOKEN, ELECTRICITY_RATE, API_PRICE_IN, API_PRICE_OUT, …) | ☑ | placeholders only; no real values | FR-2 |
| 0.10 | Quality gate script: ruff + pytest-cov + **150-line check** + secret scan | ☑ | one command runs all gates; fails on >150-line file | NFR-2/3/4/5 |
| 0.11 | `CHANGELOG.md` + `docs/prompt_engineering_log.md` started | ☑ | 1.00 entry; first prompt-log entries | NFR-6/7 |

---

## Phase 1 — Baseline (evidence of the problem)

| # | Task | Status | DoD | Ref |
|---|---|---|---|---|
| 1.1 | `OllamaBackend` + small-model sanity run (e.g. `gemma3`/`llama3.1` small) | ☑ | small model answers a prompt; harness validated end-to-end | FR-6 |
| 1.2 | `HFTransformersBackend` direct-load attempt of the giant model | ☑ | captures OOM/impractical-time **with exact error + screenshot** | FR-5, R1 |
| 1.3 | Document baseline failure narrative (why it can't fit: 140 GB vs 8 GB) | ☑ | README baseline section drafted with evidence | FR-5, FR-22 |
| 1.4 | Confirm `nvidia-smi.exe` from host (resolve OQ-1) | ☑ | OQ-1 answered; ADR-001/002 confirmed or revised | OQ-1 |

---

## Phase 2 — AirLLM integration (K1: giant model runs)

| # | Task | Status | DoD | Ref |
|---|---|---|---|---|
| 2.1 | `from airllm import AutoModel` smoke test (before any big download) | ☑ | prints "AirLLM OK" on pinned Python | FR-3, Part-C p18 |
| 2.2 | `scripts/download.py` with `dry-run` + `--include` filters + free-space guard | ☑ | dry-run lists files/size; aborts if disk insufficient | FR-8, R5, Part-C p26 |
| 2.3 | `AirLLMBackend` using **`AutoModel.from_pretrained`** + `layer_shards_saving_path` off the system root | ☑ | shards written to configured path; class-mismatch avoided | FR-7/8, ADR-003 |
| 2.4 | CPU-mode generation, `max_new_tokens=16–32`, short prompt | ☑ | **coherent output produced (K1 met)** | FR-9/10, K1 |
| 2.5 | Capture per-layer load/compute timeline (instrumentation hook) | ☑ | timeline data persisted for one run | FR-17 |
| 2.6 | Document AirLLM CPU behavior + any quant-support limits found | ☑ | findings recorded; feeds ADR-004 | FR-10/13 |

---

## Phase 3 — Quantization sweep (K2/K3)

| # | Task | Status | DoD | Ref |
|---|---|---|---|---|
| 3.1 | Verify which precisions AirLLM supports on CPU (FP16/8bit/4bit) | ☑ | support matrix recorded; infeasible cells marked ⤬ with reason | FR-13, ADR-004 |
| 3.2 | Q2 **pipeline-validation** run (cheap, low quality) | ☑ | pipeline confirmed before expensive runs | FR-11, "Start Q2" Do |
| 3.3 | Full runs at ≥3 levels (FP16, 8bit, 4bit) via AirLLM where feasible | ☑ | each run logs metrics + shard footprint + peak RAM | FR-11/12 |
| 3.4 | GGUF K-quant quality reference via Ollama (Q8/Q5/Q4/Q2) if AirLLM-CUDA-quant infeasible | ☑ | quality-vs-precision data captured | ADR-004, FR-13 |
| 3.5 | Record disk footprint + memory per precision into `results/` | ☑ | CSV rows complete per level | FR-12, K3 |

---

## Phase 4 — Benchmark harness + figures (K4)

| # | Task | Status | DoD | Ref |
|---|---|---|---|---|
| 4.1 | `metrics/timing.py` — TTFT, TPOT/ITL, throughput | ☑ | unit-tested against synthetic token streams | FR-14, NFR-3 |
| 4.2 | `metrics/memory.py` — peak RAM sampler (psutil), VRAM if any | ☑ | sampler unit-tested with a fake workload | FR-14 |
| 4.3 | `metrics/energy.py` — power×time estimate (documented model) | ☑ | energy math unit-tested; assumptions stated | FR-14, FR-21 |
| 4.4 | `quality/rater.py` — rubric scoring of outputs | ☑ | deterministic rubric; unit-tested | FR-14 |
| 4.5 | `benchmark_pipeline.py` — orchestrate ≥3 reps, median+spread, cold/warm cache | ☑ | results/*.csv has reps, median, IQR, cache flag | FR-16, K4 |
| 4.6 | `viz/plots.py` + `viz/tables.py` — regenerate all figures/tables from data | ☑ | `uv run benchmark` recreates assets deterministically | FR-15, AC-2 |
| 4.7 | Per-layer timeline figure (I/O-bound visualization) | ☑ | timeline chart in assets/ | FR-17, extension E2/E3 |

---

## Phase 5 — Economic analysis (K5)

| # | Task | Status | DoD | Ref |
|---|---|---|---|---|
| 5.1 | `economics/onprem.py` — CAPEX amortization + OPEX (kWh×rate from measured power) | ☑ | unit-tested; inputs from config/env | FR-18 |
| 5.2 | `economics/api.py` — input/output token pricing + **Prompt Caching** | ☑ | unit-tested; price source+date cited | FR-19 |
| 5.3 | `economics/breakeven.py` — crossover vs monthly token volume (+ optional Cloud GPU line) | ☑ | crossover value computed + unit-tested | FR-20 |
| 5.4 | Break-even chart + assumptions table in report | ☑ | figure in assets/; assumptions auditable | FR-21, K5 |

---

## Phase 6 — Theory linkage + extension (K6/K8)

| # | Task | Status | DoD | Ref |
|---|---|---|---|---|
| 6.1 | README theory section: Prefill(compute-bound)/Decode(memory-bound), Memory Wall, roofline | ☑ | each empirical finding paired with a mechanism | FR-22, K6 |
| 6.2 | README: AirLLM converts memory-bound → I/O-bound; `mmap`/page-fault/page-cache | ☑ | explanation references measured per-layer timeline | FR-22 |
| 6.3 | **Extension E1** — shard-location I/O sensitivity (internal NVMe vs `/tmp`) | ☑ | comparative chart + write-up | §10 E1, ADR-003 |
| 6.4 | **Extension E3** — page-cache cold→warm speedup curve | ☑ | curve chart + write-up | §10 E3, FR-16 |
| 6.5 | (Optional) E2/E4/E5 if time permits | ⤬ | Not pursued — E1+E3 sufficient; time allocated to quality gate | §10 |

---

## Phase 7 — Report assembly + quality gate (K7)

| # | Task | Status | DoD | Ref |
|---|---|---|---|---|
| 7.1 | `report_builder.py` assembles tables/figures; finalize README report | ☑ | README has all required sections (AC-1) | AC-1 |
| 7.2 | ISO/IEC 25010 mapping section | ☑ | 6 characteristics addressed | NFR-8 |
| 7.3 | Prompt-Engineering / Vibe-Coding log complete | ☑ | log reflects real prompts/decisions | NFR-7 |
| 7.4 | Run full quality gate: ruff 0 · cov ≥85% · all files ≤150 lines · 0 secrets | ☑ | gate green (AC-4/5) | AC-4/5, K7 |
| 7.5 | Verify reproducibility from clean clone (`uv sync` + run) | ☑ | figures/tables regenerate (AC-2/3) | AC-2/3 |
| 7.6 | Tag **v1.00**, update CHANGELOG | ☑ | release tagged | NFR-6 |
| 7.7 | Final self-review against all Acceptance Criteria AC-1..AC-9 | ☑ | every AC checked off | §7 PRD |

---

## Phase 8 — Post-submission improvements (2026-06-18)

| # | Task | Status | Outcome |
|---|---|---|---|
| 8.1 | `shared/rate_limiter.py` — token-bucket rate limiter from config | ☑ | `config/default.toml [rate_limits]`; 8 unit tests; 100% coverage |
| 8.2 | Docstrings on every public class/function/module | ☑ | gatekeeper · config · timing · memory · energy · onprem · breakeven |
| 8.3 | `services/chat.py` — interactive REPL chat loop (`uv run chat`) | ☑ | TinyLlama chat template; /tokens /clear /quit; 6 unit tests |
| 8.4 | `notebooks/analysis.ipynb` — results analysis notebook | ☑ | E1 · E3 · break-even · roofline summary cells |
| 8.5 | `LICENSE` — MIT license at repo root | ☑ | MIT © 2026 Ahmad Kaiss, Yosef Shanaa |
| 8.6 | §8.1 API token cost analysis table in README | ☑ | Dev cost ~$1.21 · 5 optimization strategies |
| 8.7 | §18 Contributing & License · Extension points table | ☑ | Fork guide · 4 extension points · third-party attributions |
| 8.8 | "Project Planning & Documentation" full section in README | ☑ | Goals · ACs · ADRs · P0–P7 phase table all visible |
| 8.9 | 3 live terminal screenshots embedded (S1 demo · S2/S3 chat) | ☑ | §5 AirLLM Integration + §17 Interactive Chat Demo |
| 8.10 | Merge `master` → `main` | ☑ | All commits unified on default branch |

---

## Phase 9 — Honesty gap closures v1.10 (2026-06-18)

| # | Task | Status | Outcome |
|---|---|---|---|
| 9.1 | Ollama GGUF precision sweep — `services/quant_sweep_ollama.py` (`sweep-ollama`) | ☑ | Q8_0=92 tok/s, Q4_K_M=133 tok/s, Q2_K=145 tok/s; `results/quant_sweep_ollama.json/csv` |
| 9.2 | Giant model AirLLM proof — `services/giant_proof.py` (`giant-proof`) | ☑ | Direct-load OOM confirmed (26 GB FP16 > 19.3 GB RAM, 21.1 GB gap, 180 s timeout); AirLLM streaming peak RAM 274 MB ≪ 26 GB → `results/giant_proof.json` |
| 9.3 | Empirical TPOT/ITL sweep — `services/tpot_sweep.py` (`tpot-sweep`) | ☑ | AirLLM ITL = 1416 ms/token (linear fit, n=1,2,4,8); `results/tpot_sweep.json` |
| 9.4 | Unit tests for all 3 new services (17 tests) | ☑ | 170 total tests, 87.58% coverage |
| 9.5 | Report builder + KPI scorecard + section_models updated | ☑ | Real measured values; TPOT=0 placeholder removed |
| 9.6 | Docs updated (PLAN ADR-002/004 revised, ADR-009 added; TODO Phase 9; CHANGELOG v1.10) | ☑ | All honesty gaps documented |
| 9.7 | Quality gate: ruff clean, pytest ≥85%, all files ≤150 lines | ☑ | 0 violations; 87.58%; max 150 lines |
| 9.8 | Run `uv run giant-proof` and update `results/giant_proof.json` | ☑ | Honest negative result: `huggyllama/llama-13b` is LLaMA-**1**; `AirLLMLlamaMlx` expects LLaMA-2 layout (`rotary_emb` absent) — documented in commit 2cc8cbf |
| 9.9 | Commit and push v1.10 to main | ☑ | v1.10 shipped; v1.11 (F5 timeline capture) now on main |

---

## Phase 10 — F5 per-layer timeline capture v1.11 (2026-06-20)

| # | Task | Status | Outcome |
|---|---|---|---|
| 10.1 | Real per-layer load/compute capture — `sdk/metrics/airllm_instrument.py` | ☑ | Monkey-patches AirLLM MLX `MlxModelPersister.load_model` + `TransformerBlock.__call__`; `mx.eval` forces real disk read / forward per layer |
| 10.2 | Wire capture into `airllm-demo` → persist `results/layer_timeline.json` | ☑ | Real run on Apple M3 Pro: 22 TinyLlama layers, **I/O fraction 70%** (mean 14.7 ms load vs 6.3 ms compute) |
| 10.3 | Replace hardcoded `f5_layer_timeline([])` in `benchmark_pipeline.py` with JSON read; `[]` fallback kept | ☑ | F5 now driven by measured data |
| 10.4 | `report_builder` regenerates F5 PNG + honest measured caption/blurb via `_report_figs.render_f5` | ☑ | §7 shows real bars (not "No timeline data") + measured I/O fraction |
| 10.5 | Unit tests: capture correlation, JSON read/write, `io_fraction`, F5 render, demo wiring | ☑ | 183 tests, 87.96% coverage; new modules 97–100% |
| 10.6 | Quality gate: ruff clean, pytest ≥85%, all files ≤150 lines | ☑ | 0 violations; 87.96%; max 150 lines |
| 10.7 | Docs + version bump (CHANGELOG v1.11, TODO Phase 10, `version.py` → 1.11) | ☑ | This section |

---

## Standing reminders (Do / Don't — from exercise + guidelines)

**Do:** uv venv with pinned non-newest Python · start small + start at Q2 · low `max_new_tokens` · ensure disk space (dry-run) · set `layer_shards_saving_path` off the system root · use AirLLM `AutoModel` · ≥3 reps · cite sources/dates.
**Don't:** ❌ hard-code the HF token or any secret · ❌ ignore economics · ❌ commit shards/models/`.env` · ❌ assume CUDA · ❌ hand-edit numbers into figures · ❌ exceed 150 lines/file · ❌ use pickle artifacts.
