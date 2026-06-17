"""Unit tests for sdk/model_loader/base.py"""

from airllm_local_lab.sdk.model_loader.base import GenerationResult


def test_generation_result_defaults():
    r = GenerationResult(text="hello world")
    assert r.text == "hello world"
    assert r.token_ids == []
    assert r.num_tokens == 0


def test_generation_result_with_tokens():
    r = GenerationResult(text="hi", token_ids=[1, 2, 3], num_tokens=3)
    assert len(r.token_ids) == 3
    assert r.num_tokens == 3
