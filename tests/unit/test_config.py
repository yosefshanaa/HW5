"""Unit tests for shared/config.py"""

import pytest

from airllm_local_lab.shared.config import AppConfig, ModelConfig, load_config


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.precision == "fp16"
    assert cfg.max_new_tokens == 32


def test_model_config_invalid_precision():
    with pytest.raises(ValueError):
        ModelConfig(precision="bfloat16")


def test_load_config_returns_appconfig():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)


def test_load_config_model_id_present():
    cfg = load_config()
    assert len(cfg.model.model_id) > 0


def test_env_override_precision(monkeypatch):
    monkeypatch.setenv("PRECISION", "4bit")
    cfg = load_config()
    assert cfg.model.precision == "4bit"


def test_env_override_max_tokens(monkeypatch):
    monkeypatch.setenv("MAX_NEW_TOKENS", "64")
    cfg = load_config()
    assert cfg.model.max_new_tokens == 64
