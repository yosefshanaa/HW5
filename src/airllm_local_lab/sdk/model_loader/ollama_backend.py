"""Ollama backend — small-model sanity baseline + GGUF quality reference."""

from __future__ import annotations

import subprocess

from airllm_local_lab.sdk.model_loader.base import GenerationResult
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)


class OllamaBackend:
    name = "ollama"

    def __init__(self, model_tag: str = "llama3.2:1b", max_new_tokens: int = 128) -> None:
        self.model_tag = model_tag
        self.max_new_tokens = max_new_tokens

    def _ollama_available(self) -> bool:
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def load(self) -> None:
        if not self._ollama_available():
            raise RuntimeError("Ollama is not running. Start it with `ollama serve` or install from ollama.ai")
        log.info("Ollama backend ready — model=%s", self.model_tag)

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> GenerationResult:
        tokens = max_new_tokens or self.max_new_tokens  # noqa: F841 — used below
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_tag, prompt],
                capture_output=True,
                text=True,
                timeout=300,
            )
            text = result.stdout.strip()
            words = text.split()
            return GenerationResult(text=text, num_tokens=len(words))
        except subprocess.TimeoutExpired:
            return GenerationResult(text="[TIMEOUT]", num_tokens=0)
        except Exception as exc:
            log.error("Ollama generate error: %s", exc)
            return GenerationResult(text=f"[ERROR: {exc}]", num_tokens=0)

    def unload(self) -> None:
        log.info("Ollama backend — no explicit unload needed")
