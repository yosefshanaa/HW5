"""Unit tests for sdk/model_loader/hf_backend.py"""

import pytest

from airllm_local_lab.sdk.model_loader.hf_backend import HFTransformersBackend


def test_generate_before_load_raises():
    backend = HFTransformersBackend(model_id="fake/model")
    with pytest.raises(RuntimeError, match="not loaded"):
        backend.generate("hello")


def test_unload_clears_model():
    backend = HFTransformersBackend(model_id="fake/model")
    backend._model = object()
    backend._tokenizer = object()
    backend.unload()
    assert backend._model is None
    assert backend._tokenizer is None


def test_load_oom_records_evidence():
    from unittest.mock import patch

    backend = HFTransformersBackend(model_id="huge/fake-70b")
    with (
        patch(
            "airllm_local_lab.sdk.model_loader.hf_backend.HFTransformersBackend.load",
            side_effect=MemoryError("out of memory"),
        ),
        pytest.raises(MemoryError),
    ):
        backend.load()
