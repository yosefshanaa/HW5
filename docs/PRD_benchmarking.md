# PRD — Mechanism: Performance Benchmarking Harness

| Field | Value |
|---|---|
| **Mechanism** | Measurement of latency, throughput, memory, energy & quality |
| **Version** | 1.00 · **Updated** 2026-06-17 |
| **Parent** | [PRD.md](./PRD.md) · **Design** [PLAN.md](./PLAN.md) |
| **Owning modules** | `services/benchmark_pipeline.py` + `sdk/metrics/*` + `sdk/quality/*` + `sdk/viz/*` |

---

## 1. What it measures & why

The report lives or dies on **trustworthy numbers**. This harness produces, for every feasible (model × precision × backend × cache-state) cell, a complete metric row, persisted as machine-readable data, then renders **tables + graphs** reproducibly (no hand-edited numbers — ADR-008).

### 1.1 Metric definitions (exercise §5.4)

| Metric | Definition | Phase it indexes |
|---|---|---|
| **TTFT** | Time to first token (prompt submit → first output token) | **Prefill** (compute-bound; builds KV cache) |
| **TPOT / ITL** | Inter-token latency = mean time between subsequent tokens | **Decode** (memory/I/O-bound) |
| **Throughput** | tokens / second (sustained) | overall |
| **Peak RAM** | max resident memory during run (psutil) | the binding constraint here |
| **Peak VRAM** | max GPU memory (if any) | n/a on this box (no CUDA) — recorded as 0/NA |
| **Total runtime** | wall-clock for the full generation | overall |
| **Energy / power** | estimated J / Wh = avg power (W) × runtime; method documented | economic model input |
| **Output quality** | rubric score (coherence, correctness, completeness) | quality axis |

---

## 2. Functional requirements

| ID | Requirement | Acceptance |
|---|---|---|
| **BF-1** | Capture all 8 metric families per feasible run | row complete or cell justified (K3/AC-9) |
| **BF-2** | **≥ 3 repetitions** per measured config; report **median + spread (IQR/std)** | reps stored individually + summarized (K4) |
| **BF-3** | Distinguish **cold-cache vs warm-cache** runs explicitly | `cache_state` column; both reported (Part-B page cache) |
| **BF-4** | Persist raw data to `results/*.csv|*.json` (committed) | reviewer can re-plot from data (AC-2) |
| **BF-5** | Generate all tables + graphs from that data via `viz/` | `uv run benchmark` recreates `assets/` deterministically |
| **BF-6** | Produce a **per-layer load/compute timeline** figure | visualizes I/O-bound nature (FR-17) |
| **BF-7** | Fix determinism: seeds, `torch.set_num_threads`, `OMP_NUM_THREADS` | comparable timings (R7) |
| **BF-8** | Energy method documented + assumptions stated (no spurious precision) | feeds [PRD_economic_analysis](./PRD_economic_analysis.md) |
| **BF-9** | Metric math unit-tested with synthetic token streams (no model needed) | coverage ≥ 85 % on `metrics/` (NFR-3) |
| **BF-10** | Quality rubric deterministic + documented | repeatable scores |

---

## 3. Harness design

```
benchmark_pipeline.run(matrix):
  for cell in matrix(model × precision × backend × cache_state):
     for rep in 1..R (≥3):
        reset/seed; (optionally drop caches for cold)         # BF-3/7
        start memory sampler (psutil) + timers                # BF-1
        stream = backend.generate(prompt, max_new_tokens)
        record TTFT at first token, ITL per token, throughput # BF-1
        stop sampler → peak RAM; runtime; energy = P̄ × t      # BF-1/8
        score = quality.rate(output)                          # BF-10
        append raw row → results/raw.csv                      # BF-4
  summarize → median + IQR per cell → results/summary.csv     # BF-2
  viz.tables(summary) + viz.plots(raw, summary) → assets/*    # BF-5/6
```

### 3.1 Figures to produce (assets/)
- **F1** Bar: peak RAM & shard footprint vs precision.
- **F2** Line: TTFT and TPOT vs precision (dual axis).
- **F3** Bar: throughput (tok/s) per config with error bars (IQR).
- **F4** Line: **quality vs precision** overlaid with **memory-saving vs precision** (the trade-off chart).
- **F5** **Per-layer timeline** (load vs compute) — the I/O-bound visualization (extension E2/E3).
- **F6** Cold→warm **page-cache speedup** curve (extension E3).
- **F7** (`/mnt/c` vs ext4 `~`) I/O-location comparison (extension E1).

---

## 4. Theory linkage (FR-22)

- **TTFT ↔ Prefill (compute-bound, GEMM):** large parallel matmul, builds the KV cache; first token is "expensive setup."
- **TPOT ↔ Decode (memory-bandwidth-bound, GEMV, Memory Wall):** weights re-read per token; *here* the read is from **disk via AirLLM**, so TPOT is dominated by **I/O Wait**, not FLOPs.
- **Roofline:** on this CPU-only, I/O-throttled box every measured point sits deep in the **bandwidth/I/O-bound** region — the harness data should show throughput insensitive to "compute" and sensitive to shard size & disk location.
- **Page cache (Part-B):** cold→warm curve (F6) is the OS page cache made measurable; ties `mmap`/page-fault theory to numbers.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| Noisy timings on a busy laptop | ≥3 reps, median+IQR (BF-2); fixed threads; quiesce background load; note variance |
| Energy estimate over-claimed | document as *estimate* with method + assumptions (BF-8); avoid false precision |
| Page cache hides cold cost | explicit cold runs (drop/avoid cache) vs warm (BF-3) |
| Figures drift from data | always regenerate from `results/` (BF-5/AC-2); never edit images |
| Long CPU runs limit rep count for 70B | reps focus on the smaller too-large model; 70B = single proof run (ADR-002) |

## 6. Done when
Every feasible cell has a complete, repeated, cold/warm-tagged metric row in committed `results/`, all figures/tables regenerate from that data, metric math is unit-tested ≥85%, and the per-layer timeline + trade-off charts exist — ready to embed in the README report.
