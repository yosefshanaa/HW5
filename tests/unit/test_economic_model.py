"""Unit tests for services/economic_model.py"""

import json

from airllm_local_lab.sdk.economics.api import ApiParams
from airllm_local_lab.sdk.economics.onprem import OnPremParams
from airllm_local_lab.services.economic_model import _energy_from_benchmark, build_assumptions_table
from airllm_local_lab.shared.gatekeeper import Gatekeeper


def test_energy_from_benchmark_missing(tmp_path):
    result = _energy_from_benchmark(tmp_path)
    assert abs(result - 0.00001) < 1e-9


def test_energy_from_benchmark_empty_json(tmp_path):
    (tmp_path / "benchmark_raw.json").write_text("[]")
    result = _energy_from_benchmark(tmp_path)
    assert abs(result - 0.00001) < 1e-9


def test_energy_from_benchmark_with_data(tmp_path):
    rows = [{"precision": "fp16", "energy_j": 3600.0}]
    (tmp_path / "benchmark_raw.json").write_text(json.dumps(rows))
    result = _energy_from_benchmark(tmp_path)
    assert result > 0


def test_build_assumptions_table_length():
    onprem = OnPremParams()
    api = ApiParams()
    gk = Gatekeeper()
    table = build_assumptions_table(onprem, api, gk)
    assert len(table) == 10
    assert all("parameter" in row for row in table)
