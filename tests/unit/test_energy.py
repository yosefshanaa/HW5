"""Unit tests for sdk/metrics/energy.py"""

import pytest

from airllm_local_lab.sdk.metrics.energy import EnergyResult, estimate_energy


def test_energy_joules():
    r = EnergyResult.from_runtime(10.0, avg_power_w=30.0)
    assert r.energy_j == pytest.approx(300.0, abs=1e-6)


def test_energy_kwh_conversion():
    r = EnergyResult.from_runtime(3600.0, avg_power_w=1000.0)
    assert r.energy_kwh == pytest.approx(1.0, abs=1e-6)


def test_energy_cost():
    r = EnergyResult.from_runtime(3600.0, avg_power_w=1000.0)
    assert r.cost_usd(0.15) == pytest.approx(0.15, abs=1e-6)


def test_estimate_energy_uses_tdp():
    r = estimate_energy(60.0, tdp_watts=30.0)
    assert r.energy_j == pytest.approx(1800.0, abs=1e-6)
    assert r.avg_power_w == 30.0
