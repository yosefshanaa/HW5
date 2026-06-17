"""AirLLM backend — layer-by-layer streaming inference (primary backend)."""

from __future__ import annotations

import time
from pathlib import Path

from airllm_local_lab.sdk.model_loader.base import GenerationResult
from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)

_LAYER_EVENTS: list[dict] = []


class AirLLMBackend:
    name = "airllm"

    def __init__(
        self,
        model_id: str,
        shards_path: str = "~/airllm_cache",
        compression: str | None = None,
        token: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.shards_path = str(Path(shards_path).expanduser())
        self.compression = compression
        self.token = token
        self.device = device
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        from airllm import AutoModel
        from transformers import AutoTokenizer

        Path(self.shards_path).mkdir(parents=True, exist_ok=True)
        log.info(
            "Loading %s via AirLLM (shards→ %s, compression=%s)", self.model_id, self.shards_path, self.compression
        )

        kwargs: dict = {
            "layer_shards_saving_path": self.shards_path,
        }
        if self.compression:
            kwargs["compression"] = self.compression
        if self.token:
            kwargs["hf_token"] = self.token

        self._model = AutoModel.from_pretrained(self.model_id, **kwargs)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.token, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        log.info("AirLLM model loaded — device=%s", self.device)

    def generate(self, prompt: str, max_new_tokens: int = 32) -> GenerationResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Not loaded — call load() first")

        _LAYER_EVENTS.clear()
        t0 = time.perf_counter()

        # On macOS, AirLLM uses the MLX backend which expects an MLX array input.
        try:
            from airllm import AirLLMLlamaMlx  # only importable on macOS

            if isinstance(self._model, AirLLMLlamaMlx):
                import mlx.core as mx

                token_ids = self._tokenizer.encode(prompt, add_special_tokens=True)
                x = mx.array([token_ids])
                output_text = self._model.generate(x, max_new_tokens=max_new_tokens)
                elapsed = time.perf_counter() - t0
                out_ids = self._tokenizer.encode(output_text, add_special_tokens=False)
                num_new = max(1, len(out_ids))
                log.info("MLX generated %d tokens in %.1f s", num_new, elapsed)
                return GenerationResult(text=output_text, token_ids=out_ids, num_tokens=num_new)
        except ImportError:
            pass

        # Standard PyTorch path (non-macOS / CUDA)
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs.input_ids
        out = self._model.generate(input_ids, max_new_tokens=max_new_tokens)
        elapsed = time.perf_counter() - t0
        text = self._tokenizer.decode(out[0], skip_special_tokens=True)
        num_new = out.shape[1] - input_ids.shape[1]
        log.info("Generated %d tokens in %.1f s (%.3f tok/s)", num_new, elapsed, num_new / elapsed if elapsed else 0)
        return GenerationResult(text=text, token_ids=out[0].tolist(), num_tokens=num_new)

    def unload(self) -> None:
        import gc

        self._model = None
        self._tokenizer = None
        gc.collect()
        log.info("AirLLM backend unloaded")
