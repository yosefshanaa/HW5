"""Unit tests for Extension E1 and E3 — mocked AirLLM."""

from unittest.mock import MagicMock, patch

from airllm_local_lab.sdk.model_loader.base import GenerationResult
from airllm_local_lab.services.extension_e1_io import _run_location
from airllm_local_lab.services.extension_e3_pagecache import COLD_RUNS, TOTAL_RUNS


def test_run_location_success(tmp_path):
    mock_result = GenerationResult(text="answer", num_tokens=1)

    with patch("airllm_local_lab.services.extension_e1_io.AirLLMBackend") as mock_cls:
        instance = MagicMock()
        instance.generate.return_value = mock_result
        mock_cls.return_value = instance

        elapsed = _run_location("fake/model", str(tmp_path), None, "prompt", 4)

    assert elapsed >= 0


def test_run_location_failure_propagates(tmp_path):
    import pytest

    with patch("airllm_local_lab.services.extension_e1_io.AirLLMBackend") as mock_cls:
        instance = MagicMock()
        instance.load.side_effect = RuntimeError("fail")
        mock_cls.return_value = instance

        with pytest.raises(RuntimeError):
            _run_location("fake/model", str(tmp_path), None, "prompt", 4)


def test_e1_main_writes_results(tmp_path, monkeypatch):
    mock_result = GenerationResult(text="answer", num_tokens=1)

    import airllm_local_lab.services.extension_e1_io as e1

    monkeypatch.setattr(e1, "RESULTS", tmp_path)
    monkeypatch.setattr(e1, "LOCATIONS", {"loc_a": str(tmp_path / "a"), "loc_b": str(tmp_path / "b")})

    with (
        patch("airllm_local_lab.services.extension_e1_io.AirLLMBackend") as mock_cls,
        patch("airllm_local_lab.services.extension_e1_io.f7_io_location"),
        patch("airllm_local_lab.services.extension_e1_io.Gatekeeper") as mock_gk,
    ):
        instance = MagicMock()
        instance.generate.return_value = mock_result
        mock_cls.return_value = instance
        mock_gk.return_value.hf_token.return_value = None

        e1.main()

    assert (tmp_path / "extension_e1.json").exists()


def test_e3_constants():
    assert TOTAL_RUNS >= COLD_RUNS
    assert COLD_RUNS >= 1


def test_e3_main_writes_results(tmp_path, monkeypatch):
    mock_result = GenerationResult(text="answer", num_tokens=1)

    import airllm_local_lab.services.extension_e3_pagecache as e3

    monkeypatch.setattr(e3, "RESULTS", tmp_path)
    monkeypatch.setattr(e3, "TOTAL_RUNS", 3)

    with (
        patch("airllm_local_lab.services.extension_e3_pagecache.AirLLMBackend") as mock_cls,
        patch("airllm_local_lab.services.extension_e3_pagecache.f6_page_cache_warmup"),
        patch("airllm_local_lab.services.extension_e3_pagecache.Gatekeeper") as mock_gk,
    ):
        instance = MagicMock()
        instance.generate.return_value = mock_result
        mock_cls.return_value = instance
        mock_gk.return_value.hf_token.return_value = None

        e3.main()

    assert (tmp_path / "extension_e3.json").exists()
