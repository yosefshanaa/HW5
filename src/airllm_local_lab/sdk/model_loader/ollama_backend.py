"""Ollama backend — small-model sanity baseline + GGUF quality reference."""

from __future__ import annotations

import json
import subprocess
import urllib.request

from airllm_local_lab.sdk.model_loader.base import GenerationResult
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)

_OLLAMA_BASE = "http://localhost:11434"


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

    def pull(self) -> None:
        log.info("Pulling Ollama model: %s", self.model_tag)
        subprocess.run(["ollama", "pull", self.model_tag], check=True, timeout=600)
        log.info("Pull complete: %s", self.model_tag)

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

    def generate_with_metrics(self, prompt: str, max_new_tokens: int | None = None) -> dict:
        """Call /api/generate (stream=false) and return response with timing fields."""
        num_predict = max_new_tokens or self.max_new_tokens
        payload = json.dumps(
            {
                "model": self.model_tag,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": num_predict},
            }
        ).encode()
        req = urllib.request.Request(
            f"{_OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        ev_ns = data.get("eval_duration", 0)
        ec = data.get("eval_count", 0)
        pr_ns = data.get("prompt_eval_duration", 0)
        return {
            "text": data.get("response", ""),
            "eval_count": ec,
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "prompt_eval_s": pr_ns / 1e9,
            "eval_s": ev_ns / 1e9,
            "total_s": data.get("total_duration", 0) / 1e9,
            "tpot_s": ev_ns / 1e9 / ec if ec else 0.0,
            "throughput_tps": ec / (ev_ns / 1e9) if ev_ns else 0.0,
        }

    def get_model_size_mb(self) -> float:
        """Return loaded model size in MB from /api/ps."""
        try:
            req = urllib.request.Request(f"{_OLLAMA_BASE}/api/ps")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            base = self.model_tag.split(":")[0]
            for m in data.get("models", []):
                if base in m.get("name", ""):
                    return m.get("size", 0) / 1e6
        except Exception as exc:
            log.debug("get_model_size_mb failed: %s", exc)
        return 0.0

    def unload(self) -> None:
        log.info("Ollama backend — no explicit unload needed")
