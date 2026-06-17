"""Phase 5: Economic analysis — On-Prem vs Managed API break-even."""

from __future__ import annotations

import json
from pathlib import Path

from airllm_local_lab.sdk.economics.api import ApiParams
from airllm_local_lab.sdk.economics.breakeven import build_curves, find_breakeven
from airllm_local_lab.sdk.economics.onprem import OnPremParams
from airllm_local_lab.sdk.viz.plots import breakeven_chart
from airllm_local_lab.sdk.viz.tables import economics_assumptions_table
from airllm_local_lab.shared.config import load_config
from airllm_local_lab.shared.gatekeeper import Gatekeeper
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)
RESULTS = Path(__file__).resolve().parents[3] / "results"
ASSETS = Path(__file__).resolve().parents[3] / "assets"


def _energy_from_benchmark(results_dir: Path) -> float:
    raw_file = results_dir / "benchmark_raw.json"
    if not raw_file.exists():
        log.warning("No benchmark_raw.json — using default energy estimate")
        return 0.00001
    rows = json.loads(raw_file.read_text())
    if not rows:
        return 0.00001
    fp16_rows = [r for r in rows if r.get("precision") == "fp16"]
    rows_to_use = fp16_rows or rows
    avg_j = sum(r.get("energy_j", 0) for r in rows_to_use) / len(rows_to_use)
    tokens = 32
    kwh_per_1k = (avg_j / tokens) * 1000 / 3_600_000
    return max(kwh_per_1k, 1e-8)


def build_assumptions_table(onprem: OnPremParams, api: ApiParams, gk: Gatekeeper) -> list[dict]:
    return [
        {
            "parameter": "Hardware CapEx",
            "value": f"${onprem.capex_usd:,.0f}",
            "source": "MacBook Pro M3 Pro — this machine",
        },
        {"parameter": "Hardware life", "value": f"{onprem.life_months} months", "source": "assumption"},
        {"parameter": "Maintenance/mo", "value": f"${onprem.maintenance_monthly:.2f}", "source": "placeholder"},
        {
            "parameter": "Electricity rate",
            "value": f"${onprem.electricity_rate:.3f}/kWh",
            "source": "US avg 2026-06-17",
        },
        {"parameter": "Energy/1k tokens", "value": f"{onprem.energy_per_1k_tokens_kwh:.2e} kWh", "source": "measured"},
        {
            "parameter": "API input price",
            "value": f"${api.price_in_per_1k:.5f}/1k tok",
            "source": "Anthropic 2026-06-17",
        },
        {
            "parameter": "API output price",
            "value": f"${api.price_out_per_1k:.5f}/1k tok",
            "source": "Anthropic 2026-06-17",
        },
        {
            "parameter": "Cache discount",
            "value": f"{api.cache_discount * 100:.0f}%",
            "source": "Anthropic Prompt Caching",
        },
        {"parameter": "Cache hit rate", "value": f"{api.cache_hit_rate * 100:.0f}%", "source": "assumption"},
        {"parameter": "Input fraction", "value": f"{api.in_out_ratio * 100:.0f}%", "source": "assumption"},
    ]


def main() -> None:
    cfg = load_config()
    gk = Gatekeeper()
    RESULTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    energy_per_1k = _energy_from_benchmark(RESULTS)

    onprem = OnPremParams(
        capex_usd=gk.hardware_capex(),
        life_months=gk.hardware_life_months(),
        maintenance_monthly=cfg.economics.maintenance_monthly,
        energy_per_1k_tokens_kwh=energy_per_1k,
        electricity_rate=gk.electricity_rate(),
    )
    api = ApiParams(
        price_in_per_1k=gk.api_prices().input_per_1k,
        price_out_per_1k=gk.api_prices().output_per_1k,
        cache_discount=gk.api_prices().cache_discount_fraction,
    )

    volumes = [v * 1_000 for v in range(0, 10_001, 100)]
    curves = build_curves(volumes, onprem, api, cloud_hourly=cfg.economics.cloud_gpu_hourly, tokens_per_hour=100)
    crossover = find_breakeven(onprem, api)

    breakeven_chart(volumes, curves, crossover)

    assumptions = build_assumptions_table(onprem, api, gk)
    md_table = economics_assumptions_table(assumptions)

    out = {
        "crossover_tokens": crossover,
        "crossover_note": f"On-Prem cheaper above {crossover:,.0f} tokens/month"
        if crossover
        else "API cheaper across full range",
        "onprem_fixed_monthly": round((onprem.capex_usd / onprem.life_months) + onprem.maintenance_monthly, 2),
        "assumptions": assumptions,
    }
    (RESULTS / "economics.json").write_text(json.dumps(out, indent=2))
    (ASSETS / "economics_assumptions.md").write_text(md_table)

    log.info("Economics complete — crossover: %s tokens/mo", crossover)
    print(f"\nBreak-even: {crossover:,.0f} tokens/month" if crossover else "\nAPI cheaper throughout this range")
