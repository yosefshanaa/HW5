"""On-Prem cost model: amortised CapEx + measured OPEX (electricity)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OnPremParams:
    """Parameters for the on-premises total-cost-of-ownership model."""

    capex_usd: float = 1999.0
    life_months: int = 36
    maintenance_monthly: float = 10.0
    energy_per_1k_tokens_kwh: float = 0.0
    electricity_rate: float = 0.15


def monthly_fixed_cost(params: OnPremParams) -> float:
    """Amortised hardware + maintenance, independent of token volume."""
    return (params.capex_usd / params.life_months) + params.maintenance_monthly


def monthly_variable_cost(tokens: float, params: OnPremParams) -> float:
    """Electricity cost for ``tokens`` tokens in a month."""
    return (params.energy_per_1k_tokens_kwh * tokens / 1000) * params.electricity_rate


def monthly_total(tokens: float, params: OnPremParams) -> float:
    """Total monthly on-prem cost (fixed + variable) for a given token volume."""
    return monthly_fixed_cost(params) + monthly_variable_cost(tokens, params)


def cost_curve(token_volumes: list[float], params: OnPremParams) -> list[float]:
    """Return a list of monthly costs corresponding to each volume in ``token_volumes``."""
    return [monthly_total(v, params) for v in token_volumes]
