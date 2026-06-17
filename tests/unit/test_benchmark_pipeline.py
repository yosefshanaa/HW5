"""Unit tests for services/benchmark_pipeline.py — mocked AirLLM."""

from unittest.mock import MagicMock, patch

from airllm_local_lab.sdk.model_loader.base import GenerationResult
from airllm_local_lab.services.benchmark_pipeline import _set_determinism, _summarise_reps, run_matrix


def test_summarise_reps_median():
    rows = [
        {
            "ttft_s": 1.0,
            "tpot_s": 0.5,
            "throughput_tps": 2.0,
            "peak_ram_mb": 500.0,
            "energy_j": 100.0,
            "quality_normalised": 0.8,
        },
        {
            "ttft_s": 2.0,
            "tpot_s": 0.6,
            "throughput_tps": 1.5,
            "peak_ram_mb": 600.0,
            "energy_j": 120.0,
            "quality_normalised": 0.7,
        },
        {
            "ttft_s": 3.0,
            "tpot_s": 0.7,
            "throughput_tps": 1.0,
            "peak_ram_mb": 700.0,
            "energy_j": 140.0,
            "quality_normalised": 0.6,
        },
    ]
    summary = _summarise_reps(rows)
    assert summary["ttft_median_s"] == 2.0
    assert summary["reps"] == 3


def test_summarise_reps_empty():
    summary = _summarise_reps([])
    assert summary["reps"] == 0


def test_set_determinism_runs():
    _set_determinism(42, 2)


def test_run_matrix_load_failure(tmp_path):
    with patch("airllm_local_lab.services.benchmark_pipeline.AirLLMBackend") as mock_cls:
        instance = MagicMock()
        instance.load.side_effect = RuntimeError("load failed")
        mock_cls.return_value = instance

        raw, summary = run_matrix(
            model_id="fake/model",
            precisions=["fp16"],
            shards_path=str(tmp_path),
            token=None,
            prompt="test",
            max_new_tokens=4,
            reps=1,
            seed=42,
            num_threads=1,
        )

    assert len(raw) == 0
    assert summary[0]["status"] == "infeasible"


def test_run_matrix_success(tmp_path):
    mock_result = GenerationResult(text="hello world token four", num_tokens=4)

    with patch("airllm_local_lab.services.benchmark_pipeline.AirLLMBackend") as mock_cls:
        instance = MagicMock()
        instance.generate.return_value = mock_result
        mock_cls.return_value = instance

        raw, summary = run_matrix(
            model_id="fake/model",
            precisions=["fp16"],
            shards_path=str(tmp_path),
            token=None,
            prompt="test",
            max_new_tokens=4,
            reps=2,
            seed=42,
            num_threads=1,
        )

    assert len(raw) == 2
    assert summary[0]["status"] == "ok"
    assert summary[0]["reps"] == 2
