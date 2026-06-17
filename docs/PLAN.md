# PLAN — Architecture, Design & ADRs

| Field | Value |
|---|---|
| **Document** | Technical Plan (architecture, diagrams, decisions) |
| **Project** | `airllm-local-lab` |
| **Version** | 1.00 |
| **Last updated** | 2026-06-17 |
| **Related** | [PRD.md](./PRD.md) · [TODO.md](./TODO.md) · per-mechanism PRDs |

This plan turns the [PRD](./PRD.md) into an implementable design: repository layout, C4-style architecture, runtime data-flow, the SDK/service/shared layering mandated by the submission guidelines, the API gatekeeper, the test/quality strategy, and the **Architecture Decision Records (ADRs)** that pin the project's hardest choices.

---

## 1. Architecture overview

The system is a **reproducible measurement lab**, not a service. It has four layers, mirroring the lecture's local-LLM stack (Hardware → Runtime → Compression → Adaptation), wrapped in an SDK:

```
┌──────────────────────────────────────────────────────────────────────┐
│  REPORT / RESULTS  (README.md report · results/*.csv|json · assets/*)  │
└──────────────────────────────────────────────────────────────────────┘
                              ▲ figures, tables
┌──────────────────────────────────────────────────────────────────────┐
│  SERVICES  (orchestration / use-cases)                                 │
│   baseline_runner · airllm_runner · quant_sweep · benchmark_pipeline   │
│   economic_model · report_builder                                      │
└──────────────────────────────────────────────────────────────────────┘
                              ▲ calls
┌──────────────────────────────────────────────────────────────────────┐
│  SDK  (stable, tested, reusable primitives)                            │
│   model_loader (AirLLM/HF/Ollama adapters) · metrics (TTFT/TPOT/…)     │
│   power_estimator · cost_calculator · quality_rater · plotting         │
└──────────────────────────────────────────────────────────────────────┘
                              ▲ uses
┌──────────────────────────────────────────────────────────────────────┐
│  SHARED  (cross-cutting)                                               │
│   gatekeeper (creds from env) · config · version · logging · preflight │
└──────────────────────────────────────────────────────────────────────┘
                              ▲ runs on
┌──────────────────────────────────────────────────────────────────────┐
│  HARDWARE/OS  i5-1235U · 7.6 GiB RAM · no CUDA · /mnt/c (9p) · WSL2    │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.1 C4 — Level 1 (System Context)

```
        ┌─────────────┐        accepts license, issues token
        │ Hugging Face│◀───────────────────────────────────────┐
        │     Hub     │  downloads weights (safetensors/GGUF)   │
        └─────────────┘                                         │
               │                                                │
               ▼                                                │
   ┌──────────────────────┐     reads $HF_TOKEN (env)   ┌───────┴───────┐
   │  airllm-local-lab    │◀────────────────────────────│   Researcher  │
   │  (this system)       │     runs pipeline, reads    │  (user)       │
   │                      │────────────────────────────▶│  report+data  │
   └──────────────────────┘                             └───────────────┘
        │            │
        ▼            ▼
   ┌────────┐   ┌──────────────┐
   │ Ollama │   │ Managed LLM  │  (price reference only, for economic model)
   │ (local)│   │ API (ref)    │
   └────────┘   └──────────────┘
```

### 1.2 C4 — Level 2 (Containers)

| Container | Responsibility | Key tech |
|---|---|---|
| **CLI / pipeline entrypoints** | Run baseline, AirLLM, sweep, benchmark, economics, report. | `uv run`, `typer`/`argparse` |
| **SDK package** | Device-agnostic primitives (load, measure, cost, plot). | Python, torch (CPU), `airllm`, `transformers` |
| **Runtime adapters** | Pluggable backends: `AirLLMBackend`, `HFTransformersBackend`, `OllamaBackend`. | Strategy pattern |
| **Results store** | Raw metrics (CSV/JSON), figures (PNG/SVG). | filesystem `results/`, `assets/` |
| **Config + secrets** | Typed config; gatekeeper reads env. | Pydantic/dataclass, `.env` (git-ignored) |

### 1.3 C4 — Level 3 (Components inside the SDK)

- `model_loader/` → `base.py` (Backend protocol), `airllm_backend.py`, `hf_backend.py`, `ollama_backend.py`.
- `metrics/` → `timing.py` (TTFT, TPOT/ITL, throughput), `memory.py` (peak RAM/VRAM sampler), `energy.py` (power×time estimate), `layer_timeline.py`.
- `quality/` → `rater.py` (rubric scoring of outputs).
- `economics/` → `onprem.py` (CAPEX+OPEX), `api.py` (token pricing + prompt caching), `breakeven.py`.
- `viz/` → `plots.py` (bar/line/break-even/timeline), `tables.py` (markdown tables from data).

> **150-line rule:** each file above is a *single* small responsibility, keeping every module ≤ 150 lines (NFR-4). If a file grows, split by concern (e.g., `timing.py` → `ttft.py` + `tpot.py`).

---

## 2. Repository layout (per submission guidelines)

```
airllm-local-lab/
├─ README.md                      # THE REPORT (figures/tables/screenshots embedded)
├─ pyproject.toml                 # uv-managed; pinned non-newest Python; ruff config
├─ uv.lock                        # locked deps
├─ .python-version                # 3.11 or 3.12 (NOT 3.14)
├─ .env-example                   # HF_TOKEN=..., ELECTRICITY_RATE=..., API_PRICE_*=...
├─ .gitignore                     # .env, .venv, results caches, **layer shards**, models
├─ .ruff.toml | [tool.ruff]       # select = E,F,W,I,N,UP,B,C4,SIM
├─ CHANGELOG.md                   # from 1.00
├─ docs/
│   ├─ PRD.md  PLAN.md  TODO.md
│   ├─ PRD_airllm.md  PRD_quantization.md  PRD_benchmarking.md  PRD_economic_analysis.md
│   └─ prompt_engineering_log.md  # Vibe-Coding / prompt log (guidelines)
├─ src/airllm_local_lab/
│   ├─ __init__.py
│   ├─ sdk/
│   │   ├─ model_loader/  metrics/  quality/  economics/  viz/
│   ├─ services/
│   │   ├─ baseline_runner.py  airllm_runner.py  quant_sweep.py
│   │   ├─ benchmark_pipeline.py  economic_model.py  report_builder.py
│   └─ shared/
│       ├─ gatekeeper.py   # reads creds from env ONLY
│       ├─ config.py       # typed config, loads config/*.toml + env overrides
│       ├─ version.py      # __version__ = "1.00"
│       ├─ preflight.py    # python/torch/cuda/disk/tokenizer checks
│       └─ logging.py
├─ tests/                         # mirrors src/; TDD; ≥85% coverage
│   ├─ unit/  integration/
├─ config/                        # default.toml, models.toml, economics.toml
├─ results/                       # raw CSV/JSON (committed)
├─ assets/                        # generated figures (committed) + screenshots
├─ notebooks/                     # exploratory; clean, no secrets
└─ scripts/                       # one-shot helpers (download w/ dry-run, etc.)
```

---

## 3. Runtime data-flow (one benchmarked generation)

```
config + gatekeeper(env HF_TOKEN)
        │
        ▼
preflight ── fail-fast if python/torch/disk/tokenizer wrong
        │
        ▼
Backend.load(model_id, precision, layer_shards_saving_path, device="cpu")
        │   (AirLLM: writes per-layer safetensors shards to cache path)
        ▼
warm/cold cache decision ──► sampler starts (RAM peak, wall clock, power model)
        │
        ▼
generate(prompt, max_new_tokens)        ┌─ AirLLM inner loop (per token):
        │                               │   for layer k in 0..N:
        │  emits token stream           │     load shard k → compute → release k
        ▼                               └─  (Decode = memory/IO-bound; Prefill = compute)
record: TTFT, TPOT/ITL, tok/s, peak RAM, runtime, energy, output text
        │
        ▼
quality_rater(output)  +  append row to results/*.csv|json
        │
        ▼
viz.plots + viz.tables ──► assets/*.png  ──► embedded in README report
```

---

## 4. The API Gatekeeper (NFR-5)

A single choke-point for all credentialed/config access. **No module reads `os.environ` directly except the gatekeeper.**

```python
# shared/gatekeeper.py  (sketch — ≤150 lines)
class Gatekeeper:
    """Single source of truth for secrets & external access. Reads env only."""
    def hf_token(self) -> str | None:
        return os.environ.get("HF_TOKEN")          # never hard-coded
    def require_hf_token(self) -> str:
        tok = self.hf_token()
        if not tok:
            raise ConfigError("HF_TOKEN missing — set it in .env (see .env-example)")
        return tok
    def electricity_rate(self) -> float: ...
    def api_prices(self) -> ApiPrices: ...
```

- `.env` is git-ignored; `.env-example` documents every key with placeholders.
- Tokens are **never** logged; logging filters redact anything matching a token pattern.
- AirLLM/HF calls receive the token via argument or a scoped login, sourced from the gatekeeper (Part-C p.20).

---

## 5. Backend strategy (device-agnostic)

```
Backend (Protocol): load(cfg) -> LoadedModel ; generate(prompt, max_new_tokens) -> Gen
   ├─ AirLLMBackend      # primary: layer-by-layer; CPU device; shards on cache path
   ├─ HFTransformersBackend  # baseline direct-load (expected to OOM → evidence)
   └─ OllamaBackend      # small-model sanity + GGUF quant quality reference
```

Device is **detected**, never assumed: `device = "cuda" if torch.cuda.is_available() else "cpu"`. On this machine it resolves to `cpu` (NFR-9). The same harness therefore runs unchanged on a future CUDA box, expanding the matrix automatically.

---

## 6. Test & quality strategy

- **TDD order:** write failing test → implement → green → refactor. Pure logic (metrics math, cost model, break-even, config, gatekeeper, parsers, plotting-data prep) is unit-tested to **≥ 85 %**.
- **Hard-to-test I/O** (actual model loads) is isolated behind backend interfaces and covered with **fakes/mocks**; a thin `@slow` integration test exercises the real small-model path.
- **Determinism:** fixed seeds; fixed thread counts (`torch.set_num_threads`, `OMP_NUM_THREADS`) so timing is comparable.
- **Gates (local + CI):** `uv run ruff check` (0 violations), `uv run ruff format --check`, `uv run pytest --cov` (≥ 85 %), a **line-count check** (fail if any `src/**/*.py` > 150 lines), and a **secret scan**.
- **ISO/IEC 25010 mapping** maintained in the report (functional suitability, performance efficiency, reliability, security, maintainability, portability).

---

## 7. Architecture Decision Records (ADRs)

### ADR-001 — Run CPU-only; treat "no CUDA GPU" as the central constraint
**Status:** Accepted · **Context:** `nvidia-smi` is absent; the i5-1235U exposes only Intel Iris Xe (no dedicated VRAM, no CUDA). AirLLM's headline path streams layers into CUDA VRAM. **Decision:** Execute on **CPU** (`device="cpu"`), and make the report's thesis the *harshest-case* memory↔time trade-off. Detect device at runtime; never hard-code CUDA. **Consequences:** (+) Most honest stress test of AirLLM's premise; runs on hardware the user owns. (−) Very slow; forces tiny token budgets and possibly a smaller "giant" model for the full sweep; some quantization kernels (CUDA-only `bitsandbytes`) may be unavailable → handled in ADR-004. **Re-open if** OQ-1 finds a host GPU via `nvidia-smi.exe`.

### ADR-002 — Two-tier model choice: a 70B "proof" + a smaller "too-large" sweep model
**Status:** Proposed · **Context:** A 70B model on CPU may take impractically long per token; but the exercise wants a model that genuinely doesn't fit. **Decision:** Use a **70B-class instruct model** (e.g., the `garage-bAInd/Platypus2-70B-instruct` style target from Part-C p.19, or a current Llama-3-70B/Qwen-72B) for a **single short "it runs" demonstration**, and a **smaller-but-still-too-large** model (e.g., a 13B/8B that exceeds 8 GB RAM in FP16) for the **full precision sweep + repetitions**. Use AirLLM **`AutoModel`** for both (avoids class mismatch, exercise §5.3). **Consequences:** Keeps K1 (giant runs) while making K2–K4 (sweep with reps) tractable. Both qualify as "too large for 8 GB." **Decision finalized after OQ-1/OQ-2.**

### ADR-003 — `layer_shards_saving_path` off the system root, with an I/O-location experiment
**Status:** Accepted · **Context:** Shards are large; `C:` must not be flooded; `/mnt/c` 9p I/O is slow. **Decision:** Default the cache to a configurable dir with free space, **not** `C:\` root; additionally **benchmark `/mnt/c` vs WSL2 ext4 `~`** (extension E1). **Consequences:** Avoids filling the system drive; turns the WSL2 I/O penalty into a measured insight (bottleneck shifted to I/O). Cache layout follows Part-C p.14.

### ADR-004 — Honest quantization matrix; pivot quality-vs-precision to GGUF/Ollama if CUDA-only quant is unavailable
**Status:** Proposed · **Context:** 8-bit/4-bit via `bitsandbytes` typically needs CUDA. On CPU some bit-widths may be infeasible inside AirLLM. **Decision:** Verify AirLLM's CPU quantization support **first**; benchmark the levels that genuinely work; for the **quality-degradation curve** across precisions, use **GGUF K-quants via Ollama** (Q8/Q5/Q4/Q2 are CPU-native there). Document any infeasible AirLLM cell with the systems reason (AC-9). **Consequences:** Preserves a real ≥3-level precision story even without CUDA; separates the *AirLLM memory* axis from the *quantization quality* axis cleanly.

### ADR-005 — Pin a non-newest Python (3.11 / 3.12) via uv
**Status:** Accepted · **Context:** System Python is 3.14.4; torch/airllm have no 3.14 wheels; exercise mandates non-newest Python. **Decision:** `uv python pin 3.12` (fallback 3.11); verify with the tiny `from airllm import AutoModel` import test (Part-C p.18) before any large download. **Consequences:** Reliable installs; reproducible lock.

### ADR-006 — uv only, no pip
**Status:** Accepted · **Context:** Guidelines forbid pip. **Decision:** All deps via `uv add` / `uv sync`; `pyproject.toml` + `uv.lock` committed; the lecture's `pip install` snippets are translated to `uv` equivalents. **Consequences:** Reproducible, guideline-compliant env.

### ADR-007 — `safetensors` over pickle
**Status:** Accepted · **Context:** Pickle allows code execution; AirLLM partial-reads `safetensors` via `mmap` (Part-C p.8). **Decision:** Prefer `safetensors`/GGUF artifacts exclusively. **Consequences:** Safer supply chain; aligns with AirLLM's `mmap` partial-load design.

### ADR-008 — Results are committed data; figures are generated, never hand-drawn
**Status:** Accepted · **Context:** Reproducibility (AC-2). **Decision:** Raw metrics → `results/*.csv|json` (committed); `viz` regenerates all `assets/` figures from that data. **Consequences:** Any reviewer can re-derive every chart; no number is unaudited.

---

## 8. Mapping to required exercise tasks (traceability)

| Exercise task | PRD FRs | Where built |
|---|---|---|
| 5.1 Hardware + model justification | FR-3, A4, ADR-002 | README §hardware; `preflight.py` |
| 5.2 Baseline direct run (fail/slow) | FR-5, FR-6 | `baseline_runner.py` |
| 5.3 AirLLM + quant (`AutoModel`, shards path) | FR-7–FR-13 | `airllm_runner.py`, `quant_sweep.py`, [PRD_airllm](./PRD_airllm.md), [PRD_quantization](./PRD_quantization.md) |
| 5.4 Metrics + tables/graphs | FR-14–FR-17 | `benchmark_pipeline.py`, `metrics/`, `viz/`, [PRD_benchmarking](./PRD_benchmarking.md) |
| 5.5 Economic On-Prem vs API | FR-18–FR-21 | `economic_model.py`, [PRD_economic_analysis](./PRD_economic_analysis.md) |
| 5.6 Theory linkage | FR-22 | README §theory |
| 5.7 Original extension | FR-23, §10 | chosen extension module + notebook |

---

## 9. Tooling & commands (uv-based)

```bash
uv python pin 3.12                 # ADR-005
uv sync                            # install from lock
uv run python -c "from airllm import AutoModel; print('AirLLM OK')"  # smoke (Part-C p18)
uv run preflight                   # device/disk/tokenizer checks
uv run baseline                    # FR-5/6  → evidence
uv run airllm-demo                 # FR-7..10 → K1
uv run sweep                       # FR-11..13
uv run benchmark                   # FR-14..17 → results/ + assets/
uv run economics                   # FR-18..21 → break-even chart
uv run report                      # assemble figures/tables into README
uv run ruff check . && uv run pytest --cov   # quality gate
```
