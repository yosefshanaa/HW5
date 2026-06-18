"""Unit tests for services/chat.py helper functions."""

from airllm_local_lab.services.chat import _build_prompt, _strip_template


def test_build_prompt_no_history():
    prompt = _build_prompt([], "Hello!")
    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
    assert "Hello!" in prompt
    assert "<|assistant|>" in prompt
    assert prompt.endswith("<|assistant|>")


def test_build_prompt_includes_history():
    history = [("What is 2+2?", "4.")]
    prompt = _build_prompt(history, "And 3+3?")
    assert "What is 2+2?" in prompt
    assert "4." in prompt
    assert "And 3+3?" in prompt


def test_build_prompt_trims_history_to_max():
    from airllm_local_lab.services.chat import _MAX_HISTORY

    history = [(f"q{i}", f"a{i}") for i in range(10)]
    prompt = _build_prompt(history[-_MAX_HISTORY:], "final")
    assert "q9" in prompt
    assert "q0" not in prompt


def test_strip_template_removes_markers():
    raw = "<|assistant|>\nHello there</s>"
    assert _strip_template(raw) == "Hello there"


def test_strip_template_clean_passthrough():
    clean = "Just a normal sentence."
    assert _strip_template(clean) == clean


def test_strip_template_multiple_markers():
    raw = "<|user|>hi</s><|assistant|>hello</s>"
    result = _strip_template(raw)
    assert "<|" not in result
    assert "</s>" not in result
