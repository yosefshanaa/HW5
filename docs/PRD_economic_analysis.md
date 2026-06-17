# PRD — Mechanism: Economic Analysis (On-Prem vs Managed API)

| Field | Value |
|---|---|
| **Mechanism** | Cost modelling & break-even (On-Prem CAPEX/OPEX vs Managed API vs optional Cloud GPU) |
| **Version** | 1.00 · **Updated** 2026-06-17 |
| **Parent** | [PRD.md](./PRD.md) · **Design** [PLAN.md](./PLAN.md) |
| **Owning module** | `services/economic_model.py` + `sdk/economics/*` |

---

## 1. What it is & why

The exercise explicitly forbids ignoring economics. Running locally is not "free" — it trades **CapEx + electricity + maintenance** against an API's **per-token price + convenience**. This mechanism turns the lecture's three-paradigm trade-off (Managed API / Cloud GPU / On-Prem — Part-A) into a **decision-grade, auditable model** that answers: *at what monthly token volume does On-Prem become cheaper than a Managed API for this workload?*

### 1.1 The three paradigms (Part-A trade-off table)

| Dimension | Managed API | Cloud GPU | On-Prem (this lab) |
|---|---|---|---|
| Privacy | Low | Medium | **High** |
| Control | Low | Medium | **High** |
| Cost shape | per-token OPEX | high running OPEX | **CapEx + low OPEX** |
| Maintenance | Low | Medium | **High** |
| VRAM constraint | provider-managed | scalable | **strict local limit** |

---

## 2. Functional requirements

| ID | Requirement | Acceptance |
|---|---|---|
| **EF-1** | **On-Prem model:** amortized CapEx (hardware ÷ useful-life months) + OPEX (energy = measured kWh × local rate) | unit-tested; inputs from config/env (FR-18) |
| **EF-2** | **Managed API model:** input-token price + output-token price; include **Prompt Caching** discount on repeated context | unit-tested; price **source + date** cited (FR-19) |
| **EF-3** | **Break-even:** cost-vs-monthly-token-volume crossover; state the crossover number in plain language | computed + unit-tested (FR-20, K5) |
| **EF-4** | (Optional) third line: **Cloud GPU rental** ($/hr × hours) | plotted if included (FR-20) |
| **EF-5** | All assumptions explicit in an **assumptions table** (utilization, hardware lifetime, electricity rate, $/token, tokens/request) | auditable (FR-21, AC-…) |
| **EF-6** | Energy input comes from the **measured power × runtime** in [PRD_benchmarking](./PRD_benchmarking.md) | traceable to real data |
| **EF-7** | Break-even **chart** rendered into `assets/` from committed data | reproducible (ADR-008) |

---

## 3. Cost model (formulas)

```
# On-Prem monthly cost (for volume V tokens/month)
onprem(V)   = CapEx / life_months                      # amortized hardware
            + (energy_per_1k_tokens_kWh * V/1000) * rate_per_kWh   # OPEX (measured)
            + maintenance_monthly                       # optional fixed term

# Managed API monthly cost
api(V)      = (price_in_per_1k  * in_tokens(V)/1000)
            + (price_out_per_1k * out_tokens(V)/1000)
            - prompt_cache_savings(V)                   # cached prefix discount

# Cloud GPU (optional)
cloudgpu(V) = gpu_hourly * hours_to_serve(V)

# Break-even volume V* : smallest V where onprem(V) <= api(V)
```

> **This-hardware caveat (honesty):** On-Prem here costs ~$0 extra CapEx (laptop already owned) but is **extremely slow** (CPU, no GPU). So the realistic economic story has **two regimes**: (a) *this laptop* = near-zero marginal cost but impractical throughput (good for research/POC only); (b) a *hypothetical proper On-Prem rig* (e.g., a 24 GB GPU workstation) = real CapEx with usable throughput. We model **both** and label which is which, so the break-even chart is not misleading.

---

## 4. Outputs (assets/)
- **E-Fig1** Break-even line chart: On-Prem vs API (vs Cloud GPU) cost across monthly token volume, crossover annotated.
- **E-Tab1** Assumptions table (every input + source + date).
- **E-Tab2** Scenario table: this-laptop regime vs proper-rig regime.

---

## 5. Theory / course linkage (FR-22)
- **CapEx vs OpEx**, **Privacy vs Convenience**, **Control vs Maintenance** (Part-A) framed as the axes a real buyer weighs.
- **Prompt Caching** (Part-A economics): repeated system/context prefixes are billed at a discount → materially shifts the API line for context-heavy workloads; modelled explicitly (EF-2).
- Connects to the **VRAM constraint** axis: the reason On-Prem is "hard" here is the same VRAM/memory gap AirLLM addresses — the economic and systems stories are one argument.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Stale/region-specific API & electricity prices | cite source + date; parameterize via config/env so it's re-runnable (EF-5) |
| "Free laptop" makes On-Prem look unbeatable | split into two regimes; label throughput impracticality (§3 caveat) |
| Ignoring maintenance/opportunity cost | include maintenance term + qualitative note |
| Cherry-picked token mix | state tokens/request + in:out ratio assumption; show sensitivity if time permits |

## 7. Done when
On-Prem and API cost functions are implemented + unit-tested, Prompt Caching is modelled, the break-even volume is computed and plotted with an annotated crossover, both economic regimes are presented, and every assumption is cited in an auditable table — ready for the README economics section.
