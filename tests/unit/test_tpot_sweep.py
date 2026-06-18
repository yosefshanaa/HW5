"""Unit tests for services/tpot_sweep.py — AirLLM backend mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from airllm_local_lab.sdk.model_loader.base import GenerationResult
from airllm_local_lab.services.tpot_sweep import _fit_tpot_ms, _median_time


def _make_backend():
    backend = MagicMock()

    def fake_generate(prompt, max_new_tokens=1):
        return GenerationResult(text="answer", num_tokens=max_new_tokens)

    backend.generate.side_effect = fake_generate
    return backend


def test_fit_tpot_ms_linear():
    data = [
        {"max_new_tokens": 1, "median_s": 1.6},
        {"max_new_tokens": 2, "median_s": 3.2},
        {"max_new_tokens": 4, "median_s": 6.4},
        {"max_new_tokens": 8, "median_s": 12.8},
    ]
    result = _fit_tpot_ms(data)
    assert abs(result["tpot_ms_linear_fit"] - 1600.0) < 50
    assert "note" in result


def test_fit_tpot_ms_with_base_overhead():
    data = [
        {"max_new_tokens": 1, "median_s": 2.0},  # 1s base + 1s per token
        {"max_new_tokens": 2, "median_s": 3.0},
        {"max_new_tokens": 4, "median_s": 5.0},
        {"max_new_tokens": 8, "median_s": 9.0},
    ]
    result = _fit_tpot_ms(data)
    assert abs(result["tpot_ms_linear_fit"] - 1000.0) < 50
    assert result["base_overhead_s"] > 0.5


def test_median_time_returns_stats():
    backend = _make_backend()
    with patch("time.perf_counter", side_effect=[0.0, 1.6, 1.6, 3.2, 3.2, 4.9]):
        result = _median_time(backend, max_new_tokens=1, reps=3)

    assert result["max_new_tokens"] == 1
    assert "median_s" in result
    assert "min_s" in result
    assert "max_s" in result
    assert len(result["times_s"]) == 3


def test_fit_tpot_single_point_no_crash():
    data = [{"max_new_tokens": 1, "median_s": 1.5}]
    result = _fit_tpot_ms(data)
    assert "tpot_ms_linear_fit" in result
    assert result["tpot_ms_linear_fit"] == 0.0  # ss_xx = 0 → 0
