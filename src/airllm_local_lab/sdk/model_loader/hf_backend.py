"""HF Transformers backend — direct load (expected to OOM on large models)."""

from __future__ import annotations

import time

from airllm_local_lab.sdk.model_loader.base import GenerationResult
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)

OOM_EVIDENCE: dict[str, str] = {}


class HFTransformersBackend:
    name = "hf_transformers"

    def __init__(self, model_id: str, token: str | None = None, device: str = "cpu") -> None:
        self.model_id = model_id
        self.token = token
        self.device = device
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.warning("Attempting DIRECT load of %s — expect OOM or very long wait.", self.model_id)
        t0 = time.perf_counter()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.token)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=self.token,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            elapsed = time.perf_counter() - t0
            log.info("Direct load succeeded in %.1f s (model likely too small to OOM)", elapsed)
        except MemoryError as exc:
            msg = f"MemoryError loading {self.model_id}: {exc}"
            log.error(msg)
            OOM_EVIDENCE[self.model_id] = msg
            raise
        except Exception as exc:
            msg = f"Error loading {self.model_id}: {exc}"
            log.error(msg)
            OOM_EVIDENCE[self.model_id] = msg
            raise

    def generate(self, prompt: str, max_new_tokens: int = 32) -> GenerationResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded — call load() first")
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(inputs.input_ids, max_new_tokens=max_new_tokens)
        text = self._tokenizer.decode(out[0], skip_special_tokens=True)
        return GenerationResult(text=text, num_tokens=max_new_tokens)

    def unload(self) -> None:
        import gc

        import torch

        self._model = None
        self._tokenizer = None
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        log.info("HF backend unloaded")
