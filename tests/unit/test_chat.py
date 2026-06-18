"""Unit tests for services/chat.py helper functions."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from airllm_local_lab.services import chat
from airllm_local_lab.services.chat import _build_prompt, _print_banner, _strip_template


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


def test_print_banner(capsys):
    _print_banner("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 50)
    out = capsys.readouterr().out
    assert "AirLLM Chat" in out
    assert "TinyLlama" in out
    assert "/tokens" in out


def _fake_backend():
    backend = MagicMock()
    backend.generate.return_value = SimpleNamespace(text="<|assistant|>\nhi there</s>", num_tokens=3)
    return backend


def test_chat_loop_full_session(capsys):
    """Drive the REPL through every command branch plus a normal generation turn."""
    backend = _fake_backend()
    inputs = iter(["/tokens 10", "/tokens bad", "", "/clear", "hello", "/quit"])
    with (
        patch.object(chat, "AirLLMBackend", return_value=backend),
        patch("builtins.input", lambda _="": next(inputs)),
    ):
        chat.chat_loop("m", "/tmp/shards", token=None, max_new_tokens=50)

    out = capsys.readouterr().out
    assert "Model ready" in out
    assert "max tokens set to 10" in out  # /tokens valid branch
    assert "Usage: /tokens" in out  # /tokens invalid branch
    assert "conversation cleared" in out  # /clear branch
    assert "hi there" in out  # generated + template-stripped reply
    backend.load.assert_called_once()
    backend.generate.assert_called_once()
    backend.unload.assert_called_once()


def test_chat_loop_eof_exits(capsys):
    backend = _fake_backend()

    def _raise(_=""):
        raise EOFError

    with (
        patch.object(chat, "AirLLMBackend", return_value=backend),
        patch("builtins.input", _raise),
    ):
        chat.chat_loop("m", "/tmp/shards", token=None)
    assert "Bye!" in capsys.readouterr().out
    backend.unload.assert_called_once()


def test_main_wires_config_and_loop():
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            sweep_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            layer_shards_path="~/airllm_cache",
        )
    )
    gk = MagicMock()
    gk.hf_token.return_value = None
    with (
        patch.object(chat, "load_config", return_value=cfg),
        patch.object(chat, "Gatekeeper", return_value=gk),
        patch.object(chat, "chat_loop") as loop,
    ):
        chat.main()
    loop.assert_called_once()
    assert loop.call_args.kwargs["model_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
