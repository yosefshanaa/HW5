"""Unit tests for scripts/download.py — mocked huggingface_hub."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from airllm_local_lab.scripts.download import _free_gb


def test_free_gb_returns_positive(tmp_path):
    result = _free_gb(str(tmp_path))
    assert result > 0


def test_free_gb_creates_dir(tmp_path):
    new_dir = tmp_path / "sub" / "cache"
    _free_gb(str(new_dir))
    assert new_dir.exists()


def test_list_model_files_mocked():
    mock_hf = MagicMock()
    mock_hf.list_repo_files.return_value = ["config.json", "model.safetensors"]
    with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
        from airllm_local_lab.scripts.download import list_model_files

        files = list_model_files("fake/model", token=None)
    assert len(files) == 2


def test_main_insufficient_disk(tmp_path):
    with (
        patch("airllm_local_lab.scripts.download._free_gb", return_value=1.0),
        patch("sys.argv", ["download", "fake/model", f"--cache-dir={tmp_path}", "--min-free-gb=50"]),
        pytest.raises(SystemExit),
    ):
        from airllm_local_lab.scripts.download import main

        main()
