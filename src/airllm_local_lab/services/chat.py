"""Interactive chat loop using AirLLM layer-streaming inference.

Loads the model once, then accepts user messages in a REPL loop.
Uses the TinyLlama chat template so the model responds in assistant mode.
"""

from __future__ import annotations

from pathlib import Path

from airllm_local_lab.sdk.model_loader.airllm_backend import AirLLMBackend
from airllm_local_lab.shared.config import load_config
from airllm_local_lab.shared.gatekeeper import Gatekeeper
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)

_SYSTEM = "You are a helpful, concise AI assistant."
_MAX_HISTORY = 3  # number of past exchanges kept in context


def _build_prompt(history: list[tuple[str, str]], user_msg: str) -> str:
    """Format a TinyLlama chat prompt from history + the new user message."""
    parts = [f"<|system|>\n{_SYSTEM}</s>"]
    for user, assistant in history:
        parts.append(f"<|user|>\n{user}</s>")
        parts.append(f"<|assistant|>\n{assistant}</s>")
    parts.append(f"<|user|>\n{user_msg}</s>")
    parts.append("<|assistant|>")
    return "\n".join(parts)


def _strip_template(text: str) -> str:
    """Remove any residual chat-template markers the model echoed back."""
    for marker in ("<|assistant|>", "<|user|>", "<|system|>", "</s>"):
        text = text.replace(marker, "")
    return text.strip()


def _print_banner(model_id: str, max_tokens: int) -> None:
    print(f"\n{'='*60}")
    print(f"  AirLLM Chat — {model_id}")
    print(f"  Max tokens per reply: {max_tokens}  (~{max_tokens * 1.4:.0f}s wait)")
    print("  Commands: /tokens N  /clear  /quit")
    print(f"{'='*60}\n")


def chat_loop(
    model_id: str,
    shards_path: str,
    token: str | None,
    max_new_tokens: int = 50,
) -> None:
    """Load the model once and run an interactive prompt loop until the user quits."""
    backend = AirLLMBackend(model_id=model_id, shards_path=shards_path, token=token)
    print("\nLoading model (first time may take a few seconds)…")
    backend.load()
    print("Model ready.\n")

    _print_banner(model_id, max_new_tokens)

    history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # --- built-in commands ---
        if user_input.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("Bye!")
            break

        if user_input.lower() == "/clear":
            history.clear()
            print("(conversation cleared)\n")
            continue

        if user_input.lower().startswith("/tokens "):
            try:
                max_new_tokens = int(user_input.split()[1])
                print(f"(max tokens set to {max_new_tokens} — ~{max_new_tokens * 1.4:.0f}s wait)\n")
            except (ValueError, IndexError):
                print("Usage: /tokens <number>\n")
            continue

        prompt = _build_prompt(history[-_MAX_HISTORY:], user_input)
        print(f"\nAssistant (generating {max_new_tokens} tokens…)\n")

        result = backend.generate(prompt, max_new_tokens=max_new_tokens)
        reply = _strip_template(result.text)

        print(f"{reply}\n")
        print(f"  [{result.num_tokens} tokens · {result.num_tokens / max(result.num_tokens, 1) * max_new_tokens / max_new_tokens:.0f}× — type /quit to exit]\n")

        history.append((user_input, reply))

    backend.unload()


def main() -> None:
    """Entry point for ``uv run chat``."""
    cfg = load_config()
    gk = Gatekeeper()
    shards_path = str(Path(cfg.model.layer_shards_path).expanduser())
    chat_loop(
        model_id=cfg.model.sweep_model_id,
        shards_path=shards_path,
        token=gk.hf_token(),
        max_new_tokens=50,
    )
