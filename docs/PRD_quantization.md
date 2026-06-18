# PRD — Mechanism: Quantization & Precision Sweep

| Field | Value |
|---|---|
| **Mechanism** | Weight quantization (FP16 → 8-bit → 4-bit → 2-bit) as the optimization lever |
| **Version** | 1.00 · **Updated** 2026-06-17 |
| **Parent** | [PRD.md](./PRD.md) · **Design** [PLAN.md](./PLAN.md) · **Pairs with** [PRD_airllm.md](./PRD_airllm.md) |
| **Owning module** | `services/quant_sweep.py` |

---

## 1. What it is & why

Quantization stores each weight in **fewer bits**: FP16(16) → Q8(8) → Q6(6) → Q5(5) → Q4(4) → Q2(2). Fewer bits ⇒ **less memory + smaller disk footprint + faster I/O**, at the cost of **rising quality risk** (Part-A quality-vs-memory curve; L08). The lecture's practical balance point is **Q5/Q4** ("Practical Local Balance"); **Q4_K_M** (GGUF K-quants) is the canonical sweet spot for personal computers; **Q2** is mostly for demos / pipeline validation.

Crucially (Part-C p.12): in AirLLM, **compression is an optimization lever, not a precondition** — AirLLM can run **FP16 without any quantization** by streaming layers; 4-bit/8-bit are an *additional* tool to shrink per-layer footprint, speed I/O, and reduce disk use. So this mechanism is **orthogonal** to AirLLM: AirLLM gives us "the model fits in time"; quantization tunes "how much memory/I/O per layer and at what quality."

---

## 2. Two distinct axes (do not conflate)

| Axis | Question it answers | How we measure |
|---|---|---|
| **A. Memory/I/O footprint** | How does precision change peak RAM, shard size, per-layer load time? | AirLLM runs at each feasible precision → `results/` |
| **B. Output quality** | How much does the *answer* degrade as bits drop? | rubric scoring across precisions on a fixed prompt set |

Keeping these separate lets us still produce a meaningful **quality-degradation curve** even if a given AirLLM bit-width is infeasible on CPU (we use GGUF/Ollama for axis B — ADR-004).

---

## 3. Functional requirements

| ID | Requirement | Acceptance |
|---|---|---|
| **QF-1** | Benchmark **≥ 3 precision levels** (min FP16, 8-bit, 4-bit) | exercise §5.3; PRD K2 |
| **QF-2** | **Q2 pipeline-validation** run first (cheap, low quality) | confirms pipeline before costly runs ("start with Q2" Do) |
| **QF-3** | Per level capture: **peak RAM, disk footprint of shards, per-layer load time** + full benchmark set | rows complete in `results/` (K3) |
| **QF-4** | Verify **CPU support matrix** for each bit-width in AirLLM/`bitsandbytes` **before** committing | infeasible levels marked ⤬ with the systems reason (AC-9) |
| **QF-5** | If CUDA-only quant blocks an AirLLM cell, use **GGUF K-quants via Ollama** (Q8/Q5/Q4/Q2) for the **quality axis** | quality curve still delivered (ADR-004) |
| **QF-6** | Use **`safetensors`/GGUF** artifacts only; never pickle | NFR-5 / ADR-007 |
| **QF-7** | Record format metadata (GGUF: metadata+tokenizer+quantized weights+runtime hints; K-quant variant e.g. `Q4_K_M`) | reproducibility |
| **QF-8** | Tie footprint numbers back to the **140 GB → fits-in-one-layer** narrative | report linkage |

---

## 4. Precision matrix (planned; cells confirmed in Phase 3)

| Precision | Bits/weight | Expected memory saving | Expected quality | Role in this project |
|---|---|---|---|---|
| **FP16** | 16 | baseline (largest) | reference | AirLLM headline (no quant needed) |
| **Q8** | 8 | ~2× smaller | near-reference | sweep level |
| **Q5** | 5 | larger saving | good | quality-axis (GGUF) |
| **Q4 / Q4_K_M** | 4 | "practical balance" | acceptable | **sweet spot** — primary comparison |
| **Q2** | 2 | maximum | degraded | pipeline validation / demo only |

> The exact feasible set on CPU is determined empirically (QF-4). Any level that cannot run under AirLLM on this CPU-only box is **documented with evidence**, and its quality point is still obtained via Ollama GGUF so the curve is complete.

---

## 5. Theory linkage (FR-22)

- **Memory saved ↑ vs quality risk ↑** is the central trade (Part-A): we plot both curves on shared x-axis (bits) so the crossover/sweet spot is visible.
- **Smaller bits → smaller shards → less I/O per layer → lower per-token latency in AirLLM**: connects quantization directly to the AirLLM I/O bottleneck (Part-C p.9/12).
- **GGUF K-quants / `Q4_K_M`**: why mixed-precision K-quants beat naive uniform quantization at the same average bit-width.
- **QLoRA context (L08/Part-A):** 4-bit NF4 base + FP16 adapters — noted as the bridge to the optional fine-tuning extension, not required here.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| `bitsandbytes` 8/4-bit needs CUDA → infeasible on CPU | confirm early (QF-4); pivot quality axis to GGUF/Ollama (QF-5) |
| Q2 output gibberish misread as a bug | Q2 is explicitly validation-only (QF-2); flag expected low quality |
| Disk blow-up from keeping all precisions' shards | clean shards between levels; record footprint then prune; cache off the system root |
| Comparing across two backends (AirLLM vs Ollama) unfairly | keep axes separate (§2); same prompt set + rubric; document backend per cell |

## 7. Done when
≥ 3 precision levels are benchmarked (memory/I/O axis) with footprints recorded, a quality-degradation curve exists across precisions, every infeasible CPU cell is justified, and all artifacts are `safetensors`/GGUF — feeding the benchmark tables/graphs and the theory section.
