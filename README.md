# AirLLM Local Lab — Running Giant LLMs on Constrained Local Hardware

> **HW5 / Assignment 05 — Deep-dive technical report.**
> Running large language models that *do not fit* in local memory, using **AirLLM** (layer-by-layer streaming), **quantization**, and rigorous **performance benchmarking** — on a GPU-less laptop.

[![Python](https://img.shields.io/badge/python-3.12-blue)](#) [![Package manager](https://img.shields.io/badge/deps-uv-purple)](#) [![Lint](https://img.shields.io/badge/lint-ruff-orange)](#) [![Status](https://img.shields.io/badge/status-in%20progress-yellow)](#) [![Version](https://img.shields.io/badge/version-1.00-green)](#)

---

## ⚠️ This README *is* the report

Per the assignment, this file is the technical report. Sections below are filled in as the work progresses; the planning documents live in [`docs/`](./docs).

## 📚 Planning documents
| Doc | Purpose |
|---|---|
| [docs/PRD.md](./docs/PRD.md) | Product requirements — goals, FRs/NFRs, KPIs, acceptance criteria, risks |
| [docs/PLAN.md](./docs/PLAN.md) | Architecture, C4 diagrams, ADRs, repo layout, quality strategy |
| [docs/TODO.md](./docs/TODO.md) | Phased backlog with Definition of Done |
| [docs/PRD_airllm.md](./docs/PRD_airllm.md) | Mechanism: AirLLM layer-streaming |
| [docs/PRD_quantization.md](./docs/PRD_quantization.md) | Mechanism: precision sweep (FP16→Q2) |
| [docs/PRD_benchmarking.md](./docs/PRD_benchmarking.md) | Mechanism: metrics harness |
| [docs/PRD_economic_analysis.md](./docs/PRD_economic_analysis.md) | Mechanism: On-Prem vs API economics |

---

## The problem in one line
A 70B model in FP16 is **~140 GB**. This machine has **7.6 GiB RAM and no GPU**. AirLLM makes it *run anyway* by streaming one transformer layer at a time from disk — **trading a memory limit for a time limit**.

## Target hardware (measured 2026-06-17)
| Component | Spec |
|---|---|
| CPU | 12th Gen Intel Core i5-1235U (10C / 12T) |
| RAM | 7.6 GiB to WSL2 (host 16 GB; raisable via `.wslconfig`) |
| GPU | **None** — no NVIDIA GPU on host; integrated Intel Iris Xe; **no CUDA** |
| Disk | ~237 GB free; shards on native ext4 (`~`) to avoid `/mnt/c` 9p penalty |
| OS / Python | Windows 11 + WSL2; Python pinned to 3.12 via uv |

> **Thesis:** a GPU-less, 8 GB machine is the *harshest honest* testbed for AirLLM — the painful latency **is** the finding.

---

## Report sections (to be completed)
- [ ] 1. Hardware documentation & model-choice justification
- [ ] 2. Baseline: direct-load failure evidence (HF / Ollama)
- [ ] 3. AirLLM integration (layer-by-layer, CPU mode)
- [ ] 4. Quantization & precision sweep (FP16 / Q8 / Q4 + GGUF quality axis)
- [ ] 5. Benchmark results — tables & graphs (TTFT, TPOT, throughput, RAM, energy, quality)
- [ ] 6. Economic analysis — On-Prem vs API break-even
- [ ] 7. Theory linkage — Prefill/Decode, Memory Wall, mmap/page-cache, I/O-bound shift
- [ ] 8. Original extension(s) & mini-evaluation
- [ ] 9. ISO/IEC 25010 quality mapping
- [ ] 10. Reproducibility & how-to-run

---

## Quickstart (will be wired up in Phase 0)
```bash
uv python pin 3.12
uv sync
cp .env-example .env        # then add your HF_TOKEN (never commit .env)
uv run preflight            # device / disk / tokenizer checks
```

## Security
No secrets are ever committed. The Hugging Face token is read from the environment via the gatekeeper (`.env`, git-ignored; see [`.env-example`](./.env-example)). Model weights, AirLLM layer shards, and caches are git-ignored.

## License & attribution
Coursework for educational purposes. Lecture material © Dr. Yoram Segal. AirLLM © its authors (lyogavin/airllm).
