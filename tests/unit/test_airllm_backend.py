"""Unit tests for sdk/model_loader/airllm_backend.py — mocked lazy imports."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from airllm_local_lab.sdk.model_loader.airllm_backend import AirLLMBackend


def test_generate_before_load_raises():
    b = AirLLMBackend(model_id="fake/model", shards_path="/tmp/fake")
    with pytest.raises(RuntimeError, match="Not loaded"):
        b.generate("hello")


def test_unload_clears_refs():
    b = AirLLMBackend(model_id="fake/model", shards_path="/tmp/fake")
    b._model = MagicMock()
    b._tokenizer = MagicMock()
    b.unload()
    assert b._model is None
    assert b._tokenizer is None


def _load_with_mocks(tmp_path, compression=None, token=None):
    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tok.eos_token = "<eos>"

    mock_airllm = MagicMock()
    mock_airllm.AutoModel.from_pretrained.return_value = mock_model
    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tok

    with patch.dict(sys.modules, {"airllm": mock_airllm, "transformers": mock_transformers}):
        b = AirLLMBackend(model_id="fake/model", shards_path=str(tmp_path), compression=compression, token=token)
        b.load()

    return b, mock_airllm, mock_tok


def test_load_sets_model_and_tokenizer(tmp_path):
    b, mock_am, mock_tok = _load_with_mocks(tmp_path)
    assert b._model is not None
    assert b._tokenizer is not None
    assert mock_tok.pad_token == "<eos>"


def test_load_with_compression(tmp_path):
    _b, mock_am, _tok = _load_with_mocks(tmp_path, compression="4bit")
    call_kwargs = mock_am.AutoModel.from_pretrained.call_args[1]
    assert call_kwargs.get("compression") == "4bit"


def test_load_with_token(tmp_path):
    _b, mock_am, _tok = _load_with_mocks(tmp_path, token="hf_fake123")
    call_kwargs = mock_am.AutoModel.from_pretrained.call_args[1]
    assert call_kwargs.get("hf_token") == "hf_fake123"
