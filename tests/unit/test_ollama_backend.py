"""Unit tests for sdk/model_loader/ollama_backend.py"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from airllm_local_lab.sdk.model_loader.ollama_backend import OllamaBackend


def test_ollama_not_available_raises():
    backend = OllamaBackend(model_tag="llama3.2:1b")
    with (
        patch("subprocess.run", side_effect=FileNotFoundError),
        pytest.raises(RuntimeError, match="Ollama is not running"),
    ):
        backend.load()


def test_ollama_generate_timeout():
    backend = OllamaBackend(model_tag="llama3.2:1b")
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ollama", 300)):
        result = backend.generate("hello")
    assert result.text == "[TIMEOUT]"
    assert result.num_tokens == 0


def test_ollama_generate_error():
    backend = OllamaBackend(model_tag="llama3.2:1b")
    with patch("subprocess.run", side_effect=RuntimeError("network error")):
        result = backend.generate("hello")
    assert "[ERROR:" in result.text


def test_ollama_generate_success():
    mock_result = MagicMock()
    mock_result.stdout = "Four"
    mock_result.returncode = 0
    backend = OllamaBackend(model_tag="llama3.2:1b")
    with patch("subprocess.run", return_value=mock_result):
        result = backend.generate("What is 2+2?", max_new_tokens=5)
    assert result.text == "Four"


def test_ollama_unload_noop():
    backend = OllamaBackend()
    backend.unload()


def test_ollama_available_check():
    backend = OllamaBackend()
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result):
        assert backend._ollama_available() is True
