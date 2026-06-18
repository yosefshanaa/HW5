"""Break-even computation: find volume where On-Prem ≤ API cost."""

from __future__ import annotations

from airllm_local_lab.sdk.economics.api import ApiParams
from airllm_local_lab.sdk.economics.api import monthly_cost as api_cost
from airllm_local_lab.sdk.economics.onprem import OnPremParams
from airllm_local_lab.sdk.economics.onprem import monthly_total as onprem_cost


def find_breakeven(
    onprem: OnPremParams,
    api: ApiParams,
    max_tokens: float = 100_000_000,
    step: float = 10_000,
) -> float | None:
    """Return the token volume at which on-prem becomes cheaper, or None."""
    v = 0.0
    while v <= max_tokens:
        if onprem_cost(v, onprem) <= api_cost(v, api):
            return v
        v += step
    return None


def build_curves(
    volumes: list[float],
    onprem: OnPremParams,
    api: ApiParams,
    cloud_hourly: float | None = None,
    tokens_per_hour: float = 100,
) -> dict[str, list[float]]:
    """Build monthly-cost curves for on-prem, API, and (optionally) cloud GPU over ``volumes``."""
    curves: dict[str, list[float]] = {
        "onprem": [onprem_cost(v, onprem) for v in volumes],
        "api": [api_cost(v, api) for v in volumes],
    }
    if cloud_hourly is not None:
        curves["cloud_gpu"] = [(v / tokens_per_hour) * cloud_hourly for v in volumes]
    return curves
