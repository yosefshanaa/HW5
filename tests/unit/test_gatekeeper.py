"""Unit tests for shared/gatekeeper.py"""

import pytest

from airllm_local_lab.shared.gatekeeper import ConfigError, Gatekeeper


def test_hf_token_missing(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    gk = Gatekeeper()
    assert gk.hf_token() is None


def test_hf_token_placeholder(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_your_token_here")
    gk = Gatekeeper()
    assert gk.hf_token() is None


def test_hf_token_real(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abcdefghijklmnopqrstuvwxyz12345")
    gk = Gatekeeper()
    assert gk.hf_token() == "hf_abcdefghijklmnopqrstuvwxyz12345"


def test_require_hf_token_raises(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    gk = Gatekeeper()
    with pytest.raises(ConfigError):
        gk.require_hf_token()


def test_electricity_rate_default(monkeypatch):
    monkeypatch.delenv("ELECTRICITY_RATE_PER_KWH", raising=False)
    gk = Gatekeeper()
    assert gk.electricity_rate() == pytest.approx(0.15, abs=1e-9)


def test_electricity_rate_env(monkeypatch):
    monkeypatch.setenv("ELECTRICITY_RATE_PER_KWH", "0.25")
    gk = Gatekeeper()
    assert gk.electricity_rate() == pytest.approx(0.25, abs=1e-9)


def test_api_prices_default(monkeypatch):
    monkeypatch.delenv("API_PRICE_IN_PER_1K", raising=False)
    monkeypatch.delenv("API_PRICE_OUT_PER_1K", raising=False)
    gk = Gatekeeper()
    prices = gk.api_prices()
    assert prices.input_per_1k > 0
    assert prices.output_per_1k > 0
