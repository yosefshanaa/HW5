"""Unit tests for services/airllm_runner.py — mocked AirLLM."""

from unittest.mock import MagicMock, patch

from airllm_local_lab.sdk.model_loader.base import GenerationResult
from airllm_local_lab.services.airllm_runner import run_airllm_demo


def test_run_airllm_demo_success(tmp_path):
    mock_result = GenerationResult(text="A transformer is a model.", num_tokens=6)

    with patch("airllm_local_lab.services.airllm_runner.AirLLMBackend") as mock_cls:
        instance = MagicMock()
        instance.generate.return_value = mock_result
        mock_cls.return_value = instance

        record = run_airllm_demo(
            model_id="fake/model",
            shards_path=str(tmp_path),
            token=None,
            max_new_tokens=6,
        )

    assert record["num_tokens"] == 6
    assert "output" in record
    assert record["total_s"] >= 0


def test_run_airllm_demo_with_compression(tmp_path):
    mock_result = GenerationResult(text="Answer.", num_tokens=1)

    with patch("airllm_local_lab.services.airllm_runner.AirLLMBackend") as mock_cls:
        instance = MagicMock()
        instance.generate.return_value = mock_result
        mock_cls.return_value = instance

        run_airllm_demo(
            model_id="fake/model",
            shards_path=str(tmp_path),
            token=None,
            compression="4bit",
        )

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs.get("compression") == "4bit"
