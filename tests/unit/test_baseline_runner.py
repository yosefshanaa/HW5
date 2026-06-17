"""Unit tests for services/baseline_runner.py — mocked backends."""

from unittest.mock import MagicMock, patch

from airllm_local_lab.sdk.model_loader.base import GenerationResult
from airllm_local_lab.services.baseline_runner import run_direct_load_attempt, run_small_sanity


def test_run_small_sanity_success():
    mock_result = GenerationResult(text="Four", num_tokens=1)

    with (
        patch("airllm_local_lab.services.baseline_runner.OllamaBackend") as mock_cls,
    ):
        instance = MagicMock()
        instance.generate.return_value = mock_result
        mock_cls.return_value = instance

        result = run_small_sanity("llama3.2:1b")

    assert result["status"] == "ok"
    assert result["output"] == "Four"


def test_run_small_sanity_failure():
    with patch("airllm_local_lab.services.baseline_runner.OllamaBackend") as mock_cls:
        instance = MagicMock()
        instance.load.side_effect = RuntimeError("Ollama not running")
        mock_cls.return_value = instance

        result = run_small_sanity("llama3.2:1b")

    assert result["status"] == "error"
    assert "error" in result


def test_run_direct_load_attempt_oom():
    with patch("airllm_local_lab.services.baseline_runner.HFTransformersBackend") as mock_cls:
        instance = MagicMock()
        instance.load.side_effect = MemoryError("OOM")
        mock_cls.return_value = instance

        result = run_direct_load_attempt("huge/model", token=None)

    assert result["status"] == "oom_or_error"
    assert "OOM" in result["error"]


def test_run_direct_load_attempt_unexpected_success():
    with patch("airllm_local_lab.services.baseline_runner.HFTransformersBackend") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance

        result = run_direct_load_attempt("small/model", token=None)

    assert result["status"] == "loaded"
