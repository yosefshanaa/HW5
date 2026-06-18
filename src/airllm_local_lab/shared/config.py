"""Typed configuration loaded from config/*.toml with env overrides."""

from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-reattr]

from pydantic import BaseModel, field_validator

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


class ModelConfig(BaseModel):
    """Model selection and inference parameters loaded from ``config/default.toml``."""

    model_id: str = "garage-bAInd/Platypus2-70B-instruct"
    sweep_model_id: str = "meta-llama/Meta-Llama-3-8B"
    small_model_id: str = "llama3.2:1b"
    precision: str = "fp16"
    max_new_tokens: int = 32
    layer_shards_path: str = "~/airllm_cache"

    @field_validator("precision")
    @classmethod
    def _valid_precision(cls, v: str) -> str:
        """Reject precision values not supported by the AirLLM / bitsandbytes backends."""
        allowed = {"fp16", "8bit", "4bit", "2bit"}
        if v not in allowed:
            raise ValueError(f"precision must be one of {allowed}")
        return v


class BenchmarkConfig(BaseModel):
    """Repetition, seeding, and prompt parameters for the benchmark pipeline."""

    repetitions: int = 3
    seed: int = 42
    num_threads: int = 4
    prompt_set: list[str] = [
        "Explain what a transformer is in two sentences.",
        "What is the capital of France?",
        "Write a haiku about memory.",
    ]


class EconomicsConfig(BaseModel):
    """On-prem cost model parameters (CapEx, electricity, cloud comparisons)."""

    hardware_capex: float = 1999.0
    hardware_life_months: int = 36
    maintenance_monthly: float = 10.0
    cloud_gpu_hourly: float = 0.60
    in_out_token_ratio: float = 0.40
    tokens_per_request: int = 500


class AppConfig(BaseModel):
    """Root configuration object aggregating all sub-configs."""

    model: ModelConfig = ModelConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    economics: EconomicsConfig = EconomicsConfig()


def _load_toml(path: Path) -> dict:
    """Read a TOML file; return an empty dict if the file does not exist."""
    if not path.exists():
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def load_config() -> AppConfig:
    """Load ``AppConfig`` from TOML files, then apply env-variable overrides."""
    data = _load_toml(_CONFIG_DIR / "default.toml")
    model_data = data.get("model", {})
    bench_data = data.get("benchmark", {})
    econ_data = data.get("economics", {})

    env_overrides: dict = {}
    if v := os.environ.get("MODEL_ID"):
        env_overrides["model_id"] = v
    if v := os.environ.get("PRECISION"):
        env_overrides["precision"] = v
    if v := os.environ.get("MAX_NEW_TOKENS"):
        env_overrides["max_new_tokens"] = int(v)
    if v := os.environ.get("LAYER_SHARDS_SAVING_PATH"):
        env_overrides["layer_shards_path"] = v

    model_data.update(env_overrides)
    return AppConfig(
        model=ModelConfig(**model_data),
        benchmark=BenchmarkConfig(**bench_data),
        economics=EconomicsConfig(**econ_data),
    )
