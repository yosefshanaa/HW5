"""Unit tests for services/giant_proof.py — subprocesses and AirLLM mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from airllm_local_lab.sdk.model_loader.base import GenerationResult
from airllm_local_lab.services.giant_proof import run_airllm_giant, run_direct_oom_attempt


def test_run_direct_oom_attempt_killed():
    """Simulate subprocess being killed (OOM killer)."""
    mock_result = MagicMock()
    mock_result.returncode = -9  # SIGKILL
    mock_result.stdout = ""
    mock_result.stderr = "Killed"

    with patch("subprocess.run", return_value=mock_result):
        result = run_direct_oom_attempt("facebook/opt-13b", 26.0)

    assert result["status"] in ("oom_killed", "oom_real", "oom_or_error")
    assert result["fp16_gb"] == 26.0
    assert "ram_total_gb" in result
    assert "gap_gb" in result


def test_run_direct_oom_attempt_memory_error():
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "MemoryError: Unable to allocate 26 GiB"

    with patch("subprocess.run", return_value=mock_result):
        result = run_direct_oom_attempt("facebook/opt-13b", 26.0)

    assert result["status"] == "oom_real"


def test_run_direct_oom_attempt_loaded_unexpectedly():
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "LOADED 5.2"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = run_direct_oom_attempt("tiny/model", 1.0)

    assert result["status"] == "loaded_unexpectedly"


def test_run_direct_oom_attempt_timeout():
    import subprocess

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 180)):
        result = run_direct_oom_attempt("facebook/opt-13b", 26.0)

    assert result["status"] == "oom_or_timeout"
    assert result["elapsed_s"] == 180.0


def test_run_airllm_giant_success():
    mock_result = GenerationResult(
        text="Large language models require memory because all weights must fit in RAM.",
        num_tokens=8,
    )

    with (
        patch("airllm_local_lab.services.giant_proof.AirLLMBackend") as mock_cls,
        patch("airllm_local_lab.services.giant_proof.MemorySampler") as mock_sampler,
    ):
        instance = MagicMock()
        instance.generate.return_value = mock_result
        mock_cls.return_value = instance

        mem_mock = MagicMock()
        mem_mock.peak_ram_mb = 900.0
        mock_sampler.return_value.stop.return_value = mem_mock

        result = run_airllm_giant("fake/model", "/tmp/shards", token=None)

    assert result["status"] == "ok"
    assert result["num_tokens"] == 8
    assert result["peak_ram_mb"] == 900.0
    assert result["throughput_tps"] > 0
    assert "note" in result


def test_run_airllm_giant_failure():
    with (
        patch("airllm_local_lab.services.giant_proof.AirLLMBackend") as mock_cls,
        patch("airllm_local_lab.services.giant_proof.MemorySampler") as mock_sampler,
    ):
        instance = MagicMock()
        instance.load.side_effect = RuntimeError("architecture mismatch")
        mock_cls.return_value = instance

        mem_mock = MagicMock()
        mem_mock.peak_ram_mb = 800.0
        mock_sampler.return_value.stop.return_value = mem_mock

        result = run_airllm_giant("fake/model", "/tmp/shards", token=None)

    assert result["status"] == "error"
    assert "architecture mismatch" in result["error"]
