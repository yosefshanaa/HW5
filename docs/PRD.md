# PRD — Assignment 05: Running Giant LLMs Locally with AirLLM, Quantization & Performance Benchmarking

| Field | Value |
|---|---|
| **Document** | Product Requirements Document (master) |
| **Project** | `airllm-local-lab` — Deep-dive technical report + reproducible SDK for running models that do not fit in local memory |
| **Course** | Local LLM Deployment / Orchestration — HW5 (ex05-AirLLM) |
| **Owner** | apexmediamind@gmail.com |
| **Version** | 1.00 |
| **Status** | ✅ Final — delivered 2026-06-18 · all 9 ACs met |
| **Last updated** | 2026-06-18 |
| **Related docs** | [PLAN.md](./PLAN.md) · [TODO.md](./TODO.md) · [PRD_airllm.md](./PRD_airllm.md) · [PRD_quantization.md](./PRD_quantization.md) · [PRD_benchmarking.md](./PRD_benchmarking.md) · [PRD_economic_analysis.md](./PRD_economic_analysis.md) |

---

## 1. Context & Background

Modern open-weight LLMs (Llama-3 70B, Qwen 72B, Llama-3.1 405B, …) are far larger than the memory available on consumer hardware. A 70B model in FP16 needs **~140 GB** of weights; a typical consumer GPU has **4–24 GB VRAM**, leaving a **gap of ~116 GB**. The naive conclusion is "you need a server GPU." The lecture material (Part-A/B/C, L08) reframes the problem:

> **The bottleneck is not memory — it is *when* the weights are present, not *whether* they fit.**

**AirLLM** operationalizes this. Instead of loading the whole model into VRAM, it stores every transformer layer on disk (as `safetensors` shards), then for each forward pass it: **loads layer *k* → computes the hidden state → releases layer *k* → loads layer *k+1*** … . Peak memory equals **one layer**, not the whole model. The price paid is **I/O latency**: weights are re-read from disk for every token. The memory constraint has been *traded* for a time constraint.

This assignment is a **deep-dive technical report** (the README *is* the report) backed by a **reproducible, engineered codebase** that:
1. Documents the exact local hardware and justifies a model too large to fit it.
2. Establishes a **baseline** (direct run via Ollama / HF Transformers) that fails or is impractical.
3. Integrates **AirLLM + quantization** to make the giant model *run* on this hardware.
4. **Measures** latency/throughput/memory/power/quality across quantization levels and presents the data in tables + graphs.
5. Performs an **economic analysis** (On-Prem CAPEX/OPEX vs. Managed API token cost vs. optional Cloud GPU) with a break-even chart.
6. **Connects every empirical result back to systems theory** (Prefill vs. Decode, compute-bound vs. memory-bound, the Memory Wall, virtual memory / paging / `mmap`).
7. Delivers **≥ 1 original extension** beyond the required tasks.

### 1.1 Target hardware (measured 2026-06-17 on this machine)

| Component | Specification | Implication |
|---|---|---|
| **CPU** | Apple **M3 Pro** (ARM64, 12-core — 6P + 6E) | No x86; no CUDA; AirLLM uses CPU mode via MLX on Apple Silicon. |
| **RAM** | **18 GB unified memory** (CPU + GPU share the same pool) | Hard ceiling for all processes. OPT-13b FP16 requires ~26 GB → gap = 8 GB → direct load fails. TinyLlama at 2.2 GB fits but AirLLM streams anyway, keeping peak RAM at one layer (~1.1 GB). |
| **GPU** | Apple M3 Pro GPU (Metal/MPS) — **no NVIDIA CUDA** | AirLLM's CUDA path unavailable; MLX backend used instead (`AirLLMLlamaMlx`). |
| **Disk** | 460 GiB NVMe (APFS), **193 GiB free** | Native NVMe (~7 GB/s rated); measured effective bandwidth 60.6 MB/s at AirLLM layer-streaming granularity (Python/mmap overhead dominates). |
| **OS** | macOS Darwin 25.2 (ARM64) | Native NVMe I/O; no cross-OS penalty. APFS metadata caching contributes to E1 results. |
| **Python** | **3.12** pinned via `uv python pin 3.12` | torch/airllm wheels available; satisfies non-newest-Python constraint. |
| **Package manager** | `uv` ✓ | Satisfies the mandatory "uv, never pip" rule. |

> **Critical honesty clause.** This is *not* a CUDA workstation. The report demonstrates AirLLM's value proposition on GPU-less Apple Silicon — the most honest stress test of the "trade memory for time" premise. The constraint shifts from VRAM to NVMe I/O, and the latency numbers (28 s TTFT) are the story.

---

## 2. Goals & Non-Goals

### 2.1 Goals
- **G1** — Prove that a model which **cannot** be loaded normally on this hardware **can** be executed end-to-end via AirLLM (even if slow), producing coherent output.
- **G2** — Quantify the **memory↔latency trade-off** with rigorous, reproducible metrics across ≥ 3 precision levels.
- **G3** — Produce a **decision-grade economic model** answering "when is On-Prem cheaper than a Managed API?" for this workload.
- **G4** — Tie every number to **systems theory** so the report explains *why*, not just *what*.
- **G5** — Ship a codebase that **passes the submission engineering bar** (uv, TDD ≥ 85 %, Ruff zero-violations, SDK architecture, gatekeeper config, no secrets, ≤ 150-line modules, versioned from 1.00).
- **G6** — Deliver **≥ 1 original extension** (see §10).

### 2.2 Non-Goals
- **NG1** — Not a production inference service. Real-time/high-throughput serving is explicitly out of scope (AirLLM is the wrong tool for it — Part-C p.11).
- **NG2** — Not training from scratch. (A *layer-wise fine-tuning* exploration is an *optional* extension, not a requirement.)
- **NG3** — Not a benchmark of GPU hardware we do not have. We will note where a CUDA GPU / NVMe would change results, but will not fabricate GPU numbers.
- **NG4** — Not a UI product. CLI + notebook + report only.

---

## 3. Personas & User Stories

| Persona | Need |
|---|---|
| **Researcher / student (primary)** | Run & study a giant model on a laptop without a $1,500 GPU. |
| **ML engineer evaluating deployment** | Decide On-Prem vs. API for a given workload using real cost/latency data. |
| **Course grader** | Verify the report covers all required tasks, is reproducible, and meets the engineering standard. |

**User stories** (`As a … I want … so that …`):
- **US-1** As a researcher, I want to run a 70B-class model on 8 GB RAM with no GPU, so that I can prototype with giant models on hardware I already own.
- **US-2** As a researcher, I want to compare FP16 vs Q8 vs Q4 on the same prompt, so that I understand what precision buys me in memory and costs me in latency/quality.
- **US-3** As an engineer, I want a break-even chart of On-Prem vs API cost vs monthly token volume, so that I can justify a hardware purchase (or not).
- **US-4** As a grader, I want one command to reproduce every table and figure, so that I can trust the results.
- **US-5** As any user, I want the HF token read from the environment, never hard-coded, so that the repo is safe to publish publicly.
- **US-6** As a researcher, I want clear logs of per-layer load time and peak memory, so that I can see the Memory Wall / I/O bottleneck empirically.

---

## 4. Functional Requirements

> IDs are referenced by [TODO.md](./TODO.md) and the per-mechanism PRDs.

### 4.1 Environment & configuration
- **FR-1** Provision an isolated env with **uv**, pinning a **non-newest Python (3.11 or 3.12)**. No `pip` in the workflow. `pyproject.toml` + `uv.lock` committed.
- **FR-2** All secrets/config via an **API Gatekeeper** that reads from environment / config only. **HF token** comes from `os.environ` (loaded from a git-ignored `.env`); a `.env-example` ships with placeholders only. **No token anywhere in code, history, notebooks, screenshots, or logs.**
- **FR-3** A **preflight check** module verifies: Python version, `torch` import + `torch.cuda.is_available()` (expected `False` here), free disk via `df`-equivalent, tokenizer pad/eos availability, and chosen cache path writable. Maps directly to Part-C p.21 troubleshooting table.
- **FR-4** A single **config** object (Pydantic/dataclass) defines: `model_id`, `precision` (`fp16|8bit|4bit`), `max_new_tokens`, `layer_shards_saving_path`, `device`, `prompt set`. Driven from `config/` files, overridable by env.

### 4.2 Baseline
- **FR-5** Attempt a **direct load** of the chosen giant model via HF Transformers *and* attempt the practical Ollama path; **capture and document the failure / impracticality** (OOM, time, or "won't fit"), with the exact error and the systems explanation. This is required evidence, not a throwaway.
- **FR-6** Establish a **small-model sanity baseline** (e.g., a 1–3B GGUF via Ollama, or `gemma3`/`llama3.1`) that *does* run, to validate the harness and provide a quality/latency reference point.

### 4.3 AirLLM integration (see [PRD_airllm.md](./PRD_airllm.md))
- **FR-7** Load the giant model with AirLLM's `AutoModel.from_pretrained`, **using `AutoModel` (not a model-class-specific constructor)** to avoid the Qwen/other class-mismatch issue (exercise §5.3).
- **FR-8** Set **`layer_shards_saving_path`** to a path on the drive with free space (here, a dedicated dir under the working tree / a chosen non-system location) to avoid flooding `C:`. Cache layout per Part-C p.14 (`…/model/layer_NN/…`, `tokenizer/`, `config/`).
- **FR-9** Run generation with **low `max_new_tokens`** (start at 16–32, per the "Start Small" guidance), confirm coherent output.
- **FR-10** Force/confirm **CPU execution path** (no CUDA) and document AirLLM behavior in CPU mode, including any compression options that are/aren't available without CUDA.

### 4.4 Quantization (see [PRD_quantization.md](./PRD_quantization.md))
- **FR-11** Run the giant model at **≥ 3 precision levels** — at minimum **FP16, 8-bit, 4-bit** (the exercise names FP16/Q8/Q4). Optionally include a **Q2** *pipeline-validation* run (cheap, low quality) per the "start with Q2 to validate the pipeline" Do.
- **FR-12** For each level capture: peak RAM, disk footprint of shards, per-layer load time, and the full benchmark metric set (§4.5).
- **FR-13** Document quantization **availability constraints** on CPU-only hardware (which bit-widths AirLLM/`bitsandbytes` actually support without CUDA) and adapt the matrix honestly if a level is infeasible.

### 4.5 Benchmarking (see [PRD_benchmarking.md](./PRD_benchmarking.md))
- **FR-14** Measure and record per run: **TTFT** (time-to-first-token), **TPOT/ITL** (inter-token latency), **throughput** (tokens/s), **peak RAM** (and VRAM if any), **total runtime**, **estimated energy/power**, and **qualitative output quality** (rubric-scored).
- **FR-15** Persist raw results to `results/` as machine-readable data (CSV/JSON) + generate **tables and graphs** (PNG/SVG into `assets/`) reproducibly from that data.
- **FR-16** Run each measured configuration **≥ 3 repetitions**, report **median + spread**, and record a **cold-cache vs warm-cache** distinction (page-cache effect — Part-B/C).
- **FR-17** Provide a **per-layer timeline** artifact (load vs compute per layer) to visualize the I/O-bound nature and the page-cache warmup.

### 4.6 Economic analysis (see [PRD_economic_analysis.md](./PRD_economic_analysis.md))
- **FR-18** Build an **On-Prem (CAPEX + OPEX)** model: hardware amortization + electricity (kWh × local rate) using measured power and runtime.
- **FR-19** Build a **Managed API** model: input/output token pricing for a comparable hosted model, **including Prompt Caching** effects.
- **FR-20** Compute and **plot the break-even point** (cost vs monthly token volume) and state the crossover clearly. Optionally include a **Cloud GPU rental** third line.
- **FR-21** State assumptions explicitly (utilization, hardware lifetime, electricity rate, $/token as of report date) so the model is auditable.

### 4.7 Theory linkage & extension
- **FR-22** A dedicated report section maps **measurements → theory**: Prefill (compute-bound, GEMM, builds KV cache, drives TTFT) vs Decode (memory-bandwidth-bound, GEMV, "Memory Wall", drives TPOT); roofline intuition; why AirLLM converts the problem into an **I/O-bound** one; `mmap`/page-fault/page-cache mechanics.
- **FR-23** Implement **≥ 1 original extension** (§10) with its own mini-evaluation.

---

## 5. Non-Functional Requirements

| ID | Requirement | Target / Acceptance |
|---|---|---|
| **NFR-1 Reproducibility** | One documented command reproduces all tables/figures from committed raw data; env is locked. | `uv run <pipeline>` + `uv.lock` pinned; seeds fixed. |
| **NFR-2 Code quality** | Ruff `select = E,F,W,I,N,UP,B,C4,SIM` with **zero** violations. | CI/local `uv run ruff check` clean. |
| **NFR-3 Testing** | TDD; **≥ 85 %** line coverage on non-trivial logic (metrics math, cost model, config/gatekeeper, parsers). | `uv run pytest --cov` ≥ 85 %. |
| **NFR-4 Modularity** | Every source file **≤ 150 lines**; SDK layering (`sdk/`, `services/`, `shared/`). | Enforced by a line-count check. |
| **NFR-5 Security** | No secrets in repo/history; `.env` git-ignored; `.env-example` placeholders; gatekeeper-mediated access; prefer `safetensors` over pickle. | Secret-scan clean; `.gitignore` covers `.env`, caches, shards. |
| **NFR-6 Versioning** | Semantic version from **1.00** tracked in `shared/version.py`; CHANGELOG. | Present and incremented. |
| **NFR-7 Documentation** | `docs/` (this PRD, PLAN, TODO, per-mechanism PRDs); README = the report with embedded figures; Prompt-Engineering / Vibe-Coding log. | All present. |
| **NFR-8 Quality model** | Map deliverable to **ISO/IEC 25010** characteristics (functional suitability, performance efficiency, reliability, maintainability, security, portability). | Section in report. |
| **NFR-9 Portability** | Code must not *assume* CUDA; must run CPU-only and detect device at runtime. | Preflight + device-agnostic paths. |
| **NFR-10 Storage hygiene** | Shards/caches/models are git-ignored and live on a chosen path, never committed, never on `C:` system root. | `.gitignore` + configurable `layer_shards_saving_path`. |

---

## 6. KPIs / Success Metrics

| KPI | Definition | Target |
|---|---|---|
| **K1 Giant-model executes** | Chosen too-large model produces ≥ 1 coherent answer via AirLLM on this hardware. | **Yes** (binary, primary success). |
| **K2 Precision sweep complete** | # precision levels fully benchmarked. | **≥ 3** (FP16/Q8/Q4). |
| **K3 Metric completeness** | All 8 metric families (§4.5) captured per feasible run. | **100 %** of feasible cells filled; infeasible cells justified. |
| **K4 Repetition rigor** | Repetitions per measured config. | **≥ 3**, median+spread reported. |
| **K5 Break-even delivered** | Economic crossover computed + plotted with stated assumptions. | **Yes**. |
| **K6 Theory linkage** | Each empirical finding paired with a named theoretical mechanism. | **100 %** of findings. |
| **K7 Engineering bar** | Coverage ≥ 85 %, Ruff 0 violations, all files ≤ 150 lines, 0 secrets. | **All pass**. |
| **K8 Extension** | Original extensions delivered + evaluated. | **≥ 1**. |

---

## 7. Acceptance Criteria (Definition of Done for the project)

The project is **Done** when **all** hold:
1. **AC-1** `README.md` is a complete technical report containing: hardware table, model-choice justification, baseline failure evidence, AirLLM integration, the precision-sweep results (tables **and** graphs), economic break-even chart, the theory-linkage section, and the extension(s) — with embedded figures/screenshots.
2. **AC-2** Every figure/table is regenerated by a committed pipeline from committed raw `results/` data (no hand-edited numbers).
3. **AC-3** `uv sync` + the documented run command works from a clean clone on comparable hardware (or degrades gracefully with a clear message where hardware differs).
4. **AC-4** `uv run ruff check` → 0 violations; `uv run pytest --cov` → ≥ 85 %; no source file > 150 lines.
5. **AC-5** No secret material anywhere; `.env-example` present; `.env` git-ignored; gatekeeper used for all credentialed access.
6. **AC-6** `docs/` contains this PRD, PLAN, TODO, and the four per-mechanism PRDs, all consistent with the code.
7. **AC-7** Version = 1.00+ in `shared/version.py`; Prompt-Engineering log present; ISO/IEC 25010 mapping present.
8. **AC-8** ≥ 3 precision levels benchmarked with ≥ 3 reps each; break-even computed; ≥ 1 extension delivered.
9. **AC-9** Where a requirement is infeasible on this hardware (e.g., a quant level needing CUDA), the report **documents the infeasibility with evidence and the systems reason**, rather than omitting it.

---

## 8. Constraints & Assumptions

**Hard constraints**
- **C1** No CUDA GPU → CPU execution; expect long runtimes. Mitigate with tiny `max_new_tokens`, short prompts, and a smaller "giant-relative-to-this-box" model where a 70B run is impractically slow.
- **C2** 7.6 GiB RAM ceiling → per-layer footprint must stay < available RAM; monitor for OOM; use swap as a safety net but record it.
- **C3** `/mnt/c` 9p I/O is slow → disk-read times will dominate and be *pessimistic* vs native NVMe; this is acknowledged and analyzed, and a native-Linux-path or alternative cache location is tested if it improves results.
- **C4** Must use `uv`, pinned non-newest Python; Ruff/test/line-limit/secret rules are non-negotiable (submission guidelines).
- **C5** Gated HF models require accepting a license + a personal token supplied via env only.

**Assumptions**
- **A1** ≥ ~50 GB free disk can be dedicated to shards for a 70B-class model (we have 237 GB).
- **A2** Network bandwidth is sufficient to download model weights once (large, one-time).
- **A3** Electricity rate and API $/token will be cited with a date and source in the economic model.
- **A4** "Giant model" is defined **relative to this hardware**: any model whose full weights exceed ~8 GB RAM qualifies; the headline target is a **70B-class** model, with a documented fallback if 70B CPU runtime is prohibitive.

---

## 9. Risks & Mitigations

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | 70B CPU run takes hours/token → can't finish | High | High | Tiny token budgets; single short prompt for the 70B headline; use a smaller giant model (e.g., 13B/8B "too large for 8 GB") for the full sweep; document 70B as a "proof it runs" point. |
| **R2** | AirLLM quantization needs CUDA/`bitsandbytes` GPU → some bit-widths infeasible | High | Medium | Verify support matrix early; pivot to GGUF-quantized comparisons via Ollama for the precision-vs-quality axis if needed; document honestly (AC-9). |
| **R3** | Python 3.14 breaks torch/airllm install | High | High | Pin 3.11/3.12 via `uv python pin`; verify with a tiny import test before downloading any model (Part-C p.18). |
| **R4** | `/mnt/c` I/O makes numbers unrepresentative | Medium | Medium | Also test shards on the WSL2 ext4 home (`~`) and compare; report both; frame as I/O-sensitivity analysis (bonus insight). |
| **R5** | Disk fills mid-download | Medium | High | `dry-run` + `--include` filters (Part-C p.26); preflight free-space check; configurable cache path off `C:`. |
| **R6** | Accidental token/secret commit | Low | Critical | `.gitignore` first commit; gatekeeper; pre-commit secret scan; never print token in logs. |
| **R7** | Non-deterministic timings | High | Medium | ≥ 3 reps, median+IQR; pin seeds; record cold/warm cache; fix thread counts. |
| **R8** | Scope creep (treating it as a serving product) | Medium | Medium | NG-1 explicit; keep to report+SDK. |

---

## 10. Original Extensions (choose ≥ 1; R8-safe)

Candidate extensions, each with a measurable mini-evaluation:
- **E1 — Shard-location I/O sensitivity study.** Benchmark identical runs with `layer_shards_saving_path` on (a) `/mnt/c` 9p vs (b) WSL2 native ext4, quantifying the I/O-bound penalty. Directly demonstrates "the bottleneck moved from VRAM to I/O" (Part-C p.6/9).
- **E2 — Prefetch / overlap experiment.** Measure effect of overlapping next-layer load with current-layer compute (Part-C p.13) — even a simple double-buffer prototype — on TTFT/TPOT.
- **E3 — Page-cache warmup curve.** Quantify cold→warm speedup across repeated runs to make the OS page cache (Part-B) visible as a measured curve.
- **E4 — Quality-vs-precision rubric.** Blind-score outputs across precision levels on a fixed prompt set to chart the quality-degradation curve against the memory-savings curve.
- **E5 — Layer-wise micro fine-tune POC.** Tiny LoRA-style adapter experiment framed by the layer-wise fine-tuning cycle (Part-C p.15) — clearly scoped as exploratory.

---

## 11. Timeline / Phasing (maps to TODO.md)

| Phase | Theme | Exit criterion |
|---|---|---|
| **P0** | Repo + env + guardrails | Structure, uv env (pinned Python), gatekeeper, `.env-example`, CI checks, preflight all green. |
| **P1** | Baseline | Direct-load failure documented; small-model sanity run passes. |
| **P2** | AirLLM integration | Giant model emits coherent output on CPU (K1). |
| **P3** | Quantization sweep | ≥ 3 precision levels benchmarked (K2/K3). |
| **P4** | Benchmark harness + figures | All metrics, ≥ 3 reps, tables+graphs reproducible (K4). |
| **P5** | Economic analysis | Break-even chart + assumptions (K5). |
| **P6** | Theory linkage + extension | Findings mapped to theory; ≥ 1 extension done (K6/K8). |
| **P7** | Report + quality gate | README report complete; engineering bar met (K7); version 1.00 tagged. |

---

## 12. Open Questions — Resolved

| OQ | Question | Resolution |
|---|---|---|
| **OQ-1** | Is a discrete GPU available? (nvidia-smi check) | ✅ Resolved — macOS ARM, no CUDA GPU. MLX backend used. ADR-001 confirmed. |
| **OQ-2** | Which exact giant model? | ✅ Resolved — `facebook/opt-13b` (analytic OOM proof, 26 GB > 18 GB); `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (live demo + sweep, LLaMA-compat with MLX). ADR-002 accepted. |
| **OQ-3** | Electricity rate + API price source/date? | ✅ Resolved — $0.15/kWh; Claude 3 Haiku $0.00025/$0.00125 per 1k tokens (Anthropic pricing 2026-06-17). Documented in `config/economics.toml`. |
| **OQ-4** | Which extensions to commit to? | ✅ Resolved — E1 (I/O sensitivity: NVMe vs /tmp) + E3 (page-cache warmup curve). Both delivered with charts and write-ups. |
