"""Unit tests for services/quant_sweep.py — mocked AirLLM."""

from unittest.mock import MagicMock, patch

from airllm_local_lab.sdk.model_loader.base import GenerationResult
from airllm_local_lab.services.quant_sweep import run_precision


def test_run_precision_success(tmp_path):
    mock_result = GenerationResult(text="A transformer is a model.", num_tokens=6)

    with patch("airllm_local_lab.services.quant_sweep.AirLLMBackend") as mock_cls:
        instance = MagicMock()
        instance.generate.return_value = mock_result
        mock_cls.return_value = instance

        row = run_precision(
            model_id="fake/model",
            precision="fp16",
            shards_path=str(tmp_path),
            token=None,
            prompt="Explain transformers.",
        )

    assert row["status"] == "ok"
    assert row["precision"] == "fp16"
    assert row["num_tokens"] == 6
    assert row["quality_normalised"] > 0


def test_run_precision_failure(tmp_path):
    with patch("airllm_local_lab.services.quant_sweep.AirLLMBackend") as mock_cls:
        instance = MagicMock()
        instance.load.side_effect = RuntimeError("CUDA unavailable")
        mock_cls.return_value = instance

        row = run_precision(
            model_id="fake/model",
            precision="8bit",
            shards_path=str(tmp_path),
            token=None,
            prompt="test",
        )

    assert row["status"] == "error"
    assert "CUDA" in row["error"]


def test_run_precision_4bit(tmp_path):
    mock_result = GenerationResult(text="short answer", num_tokens=2)

    with patch("airllm_local_lab.services.quant_sweep.AirLLMBackend") as mock_cls:
        instance = MagicMock()
        instance.generate.return_value = mock_result
        mock_cls.return_value = instance

        row = run_precision(
            model_id="fake/model",
            precision="4bit",
            shards_path=str(tmp_path),
            token=None,
            prompt="test",
        )

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs.get("compression") == "4bit"
    assert row["precision"] == "4bit"
