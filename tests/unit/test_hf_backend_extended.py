"""Extended tests for hf_backend.py — mock torch and transformers."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from airllm_local_lab.sdk.model_loader.hf_backend import HFTransformersBackend


def _make_loaded_backend(model_id: str = "fake/small"):
    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_transformers = MagicMock()
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tok

    import torch

    mock_out = torch.tensor([[1, 2, 3]])
    mock_model.generate.return_value = mock_out
    mock_tok.decode.return_value = "The answer is 42."
    mock_tok.return_value = MagicMock(input_ids=torch.tensor([[1, 2]]))

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        b = HFTransformersBackend(model_id=model_id)
        b.load()

    b._model = mock_model
    b._tokenizer = mock_tok
    return b


def test_load_sets_attributes():
    b = _make_loaded_backend()
    assert b._model is not None
    assert b._tokenizer is not None


def test_generate_calls_model():
    b = _make_loaded_backend()
    import torch

    b._tokenizer.return_value = MagicMock(input_ids=torch.tensor([[1, 2]]))
    b.generate("hello", max_new_tokens=3)
    b._model.generate.assert_called_once()


def test_load_captures_memory_error():
    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.side_effect = MemoryError("OOM")

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        b = HFTransformersBackend(model_id="huge/model")
        with pytest.raises(MemoryError):
            b.load()
