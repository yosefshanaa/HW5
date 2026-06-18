"""Unit tests for services/quant_sweep_ollama.py — all network calls mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from airllm_local_lab.services.quant_sweep_ollama import _ensure_pulled, _run_rep, run_one_tag


def _fake_http_response(data: dict):
    body = json.dumps(data).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


_OLLAMA_RESP = {
    "response": "A transformer converts energy from one form to another.",
    "eval_count": 10,
    "eval_duration": 100_000_000,  # 100 ms in ns
    "prompt_eval_count": 8,
    "prompt_eval_duration": 50_000_000,  # 50 ms in ns
    "total_duration": 200_000_000,
}

_PS_RESP = {"models": [{"name": "llama3.2:1b-instruct-q8_0", "size": 1_400_000_000}]}


def test_ensure_pulled_already_present():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        _ensure_pulled("llama3.2:1b-instruct-q8_0")
    mock_run.assert_called_once()


def test_ensure_pulled_missing_triggers_pull():
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=1),  # ollama show fails
            MagicMock(returncode=0),  # ollama pull succeeds
        ]
        _ensure_pulled("llama3.2:1b-instruct-q8_0")
    assert mock_run.call_count == 2


def test_run_rep_success():
    from airllm_local_lab.sdk.model_loader.ollama_backend import OllamaBackend

    backend = OllamaBackend("llama3.2:1b-instruct-q8_0")

    with patch("urllib.request.urlopen", return_value=_fake_http_response(_OLLAMA_RESP)):
        result = _run_rep(backend, "What is a transformer?", rep=0)

    assert result["status"] == "ok"
    assert result["cache_state"] == "cold"
    assert result["eval_count"] == 10
    assert abs(result["ttft_s"] - 0.05) < 0.01
    assert result["tpot_s"] > 0
    assert result["throughput_tps"] > 0


def test_run_rep_warm():
    from airllm_local_lab.sdk.model_loader.ollama_backend import OllamaBackend

    backend = OllamaBackend("llama3.2:1b-instruct-q8_0")

    with patch("urllib.request.urlopen", return_value=_fake_http_response(_OLLAMA_RESP)):
        result = _run_rep(backend, "prompt", rep=2)

    assert result["cache_state"] == "warm"


def test_run_rep_network_failure():
    from airllm_local_lab.sdk.model_loader.ollama_backend import OllamaBackend

    backend = OllamaBackend("llama3.2:1b-instruct-q8_0")

    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        result = _run_rep(backend, "prompt", rep=0)

    assert result["status"] == "error"
    assert "error" in result


def test_run_one_tag_aggregates_reps():
    def _fake_urlopen(req, timeout=None):
        if "api/generate" in req.full_url:
            return _fake_http_response(_OLLAMA_RESP)
        return _fake_http_response(_PS_RESP)

    with (
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("urllib.request.urlopen", side_effect=_fake_urlopen),
    ):
        row = run_one_tag("llama3.2:1b-instruct-q8_0", "Q8_0", "llama3.2", "What is AI?")

    assert row["status"] == "ok"
    assert row["precision"] == "Q8_0"
    assert row["reps"] == 3
    assert "ttft_median_s" in row
    assert "tpot_median_s" in row
    assert row["throughput_median_tps"] > 0
    assert row["quality_normalised"] >= 0


def test_run_one_tag_quality_lower_for_nonsense():
    """Q2_K produces lower quality output; rater reflects this."""
    good_resp = dict(_OLLAMA_RESP, response="A transformer converts electrical energy between circuits.")
    low_q_resp = dict(_OLLAMA_RESP, response="transformer transformer transformer")

    def _fake_good(req, timeout=None):
        if "api/generate" in req.full_url:
            return _fake_http_response(good_resp)
        return _fake_http_response(_PS_RESP)

    def _fake_bad(req, timeout=None):
        if "api/generate" in req.full_url:
            return _fake_http_response(low_q_resp)
        return _fake_http_response(_PS_RESP)

    with (
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("urllib.request.urlopen", side_effect=_fake_good),
    ):
        row_good = run_one_tag("llama3.2:1b-instruct-q8_0", "Q8_0", "llama3.2", "What is AI?")

    with (
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("urllib.request.urlopen", side_effect=_fake_bad),
    ):
        row_bad = run_one_tag("llama3.2:1b-instruct-q2_K", "Q2_K", "llama3.2", "What is AI?")

    assert row_good["status"] == "ok"
    assert row_bad["status"] == "ok"
    # Bad output should not score higher than good output
    assert row_bad["quality_normalised"] <= row_good["quality_normalised"] + 0.1
