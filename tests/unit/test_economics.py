"""Unit tests for sdk/economics/ modules."""

import pytest

from airllm_local_lab.sdk.economics.api import ApiParams, cost_curve, monthly_cost
from airllm_local_lab.sdk.economics.breakeven import build_curves, find_breakeven
from airllm_local_lab.sdk.economics.onprem import OnPremParams, monthly_fixed_cost, monthly_total, monthly_variable_cost

# ── OnPrem ──────────────────────────────────────────────────────────────────


def test_onprem_fixed_cost():
    p = OnPremParams(capex_usd=1200.0, life_months=36, maintenance_monthly=10.0)
    assert monthly_fixed_cost(p) == pytest.approx(43.33, abs=0.01)


def test_onprem_zero_volume():
    p = OnPremParams()
    assert monthly_variable_cost(0.0, p) == 0.0


def test_onprem_total_is_fixed_plus_variable():
    p = OnPremParams(
        capex_usd=360.0, life_months=36, maintenance_monthly=0.0, energy_per_1k_tokens_kwh=1.0, electricity_rate=0.10
    )
    fixed = monthly_fixed_cost(p)
    variable = monthly_variable_cost(1000.0, p)
    assert monthly_total(1000.0, p) == pytest.approx(fixed + variable, abs=1e-9)


# ── API ─────────────────────────────────────────────────────────────────────


def test_api_cost_zero():
    p = ApiParams()
    assert monthly_cost(0.0, p) == 0.0


def test_api_cost_positive():
    p = ApiParams(price_in_per_1k=0.001, price_out_per_1k=0.002)
    assert monthly_cost(10_000.0, p) > 0


def test_api_cost_curve_length():
    p = ApiParams()
    volumes = [0, 1000, 2000]
    assert len(cost_curve(volumes, p)) == 3


def test_cache_discount_reduces_cost():
    no_cache = ApiParams(price_in_per_1k=0.001, price_out_per_1k=0.002, cache_discount=0.0, cache_hit_rate=0.5)
    with_cache = ApiParams(price_in_per_1k=0.001, price_out_per_1k=0.002, cache_discount=0.9, cache_hit_rate=0.5)
    assert monthly_cost(10_000.0, with_cache) < monthly_cost(10_000.0, no_cache)


# ── Break-even ───────────────────────────────────────────────────────────────


def test_breakeven_found():
    # onprem fixed at $50/mo, API grows at $0.01/1k tokens → crossover near 5M tokens
    onprem = OnPremParams(capex_usd=0.0, life_months=1, maintenance_monthly=50.0, energy_per_1k_tokens_kwh=0.0)
    api = ApiParams(price_in_per_1k=0.01, price_out_per_1k=0.01, cache_discount=0.0, cache_hit_rate=0.0)
    v = find_breakeven(onprem, api, max_tokens=10_000_000, step=10_000)
    assert v is not None
    assert v > 0


def test_breakeven_none_when_api_always_cheaper():
    onprem = OnPremParams(capex_usd=1_000_000.0, life_months=1, maintenance_monthly=0.0)
    api = ApiParams(price_in_per_1k=0.0, price_out_per_1k=0.0)
    v = find_breakeven(onprem, api, max_tokens=100, step=10)
    assert v is None


def test_build_curves_keys():
    onprem = OnPremParams()
    api = ApiParams()
    curves = build_curves([0, 1000], onprem, api)
    assert "onprem" in curves
    assert "api" in curves


def test_build_curves_with_cloud():
    onprem = OnPremParams()
    api = ApiParams()
    curves = build_curves([0, 1000], onprem, api, cloud_hourly=2.0)
    assert "cloud_gpu" in curves
