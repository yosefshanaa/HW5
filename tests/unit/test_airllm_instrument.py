"""Unit tests for sdk/metrics/airllm_instrument.py — capture + persistence."""

import json
import time

from airllm_local_lab.sdk.metrics.airllm_instrument import (
    _Capture,
    capture_layer_timeline,
    io_fraction_of,
    read_timeline,
    write_timeline,
)
from airllm_local_lab.sdk.metrics.layer_timeline import LayerTimeline


def _noop(_obj):
    pass


def test_capture_correlates_load_with_next_forward():
    tl = LayerTimeline()
    cap = _Capture(tl, _noop)
    weights = cap.on_load("model.layers.3", lambda: {"w": 1})
    assert weights == {"w": 1}
    cap.on_forward(("hidden", "cache"))
    assert len(tl.events) == 1
    assert tl.events[0].layer_idx == 3


def test_capture_skips_non_layer_shards():
    tl = LayerTimeline()
    cap = _Capture(tl, _noop)
    cap.on_load("model.embed_tokens", lambda: {"e": 1})
    cap.on_forward(("hidden", "cache"))  # no pending layer → nothing recorded
    assert tl.events == []


def test_capture_records_each_layer_only_once():
    tl = LayerTimeline()
    cap = _Capture(tl, _noop)
    cap.on_load("model.layers.0", lambda: {})
    cap.on_forward(("h", "c"))
    # decode pass re-loads/re-runs the same layer; must not double-record
    cap.on_load("model.layers.0", lambda: {})
    cap.on_forward(("h", "c"))
    assert len(tl.events) == 1


def test_capture_measures_real_load_time():
    tl = LayerTimeline()
    cap = _Capture(tl, _noop)

    def slow_load():
        time.sleep(0.005)
        return {}

    cap.on_load("model.layers.7", slow_load)
    cap.on_forward(("h", "c"))
    assert tl.events[0].load_ms > 0


def test_write_read_roundtrip(tmp_path):
    dicts = [{"layer": 0, "load_ms": 10.0, "compute_ms": 2.0, "total_ms": 12.0}]
    out = tmp_path / "results" / "layer_timeline.json"
    write_timeline(dicts, out)
    assert out.exists()
    assert read_timeline(out) == dicts


def test_read_missing_returns_empty(tmp_path):
    assert read_timeline(tmp_path / "absent.json") == []


def test_read_invalid_shape_returns_empty(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"not": "a list"}))
    assert read_timeline(f) == []


def test_io_fraction_of_known_values():
    dicts = [{"load_ms": 80.0, "compute_ms": 20.0}, {"load_ms": 80.0, "compute_ms": 20.0}]
    assert io_fraction_of(dicts) == 0.8


def test_io_fraction_of_empty():
    assert io_fraction_of([]) == 0.0


def test_capture_context_manager_yields_and_restores():
    """On MLX hosts the real classes are patched then restored; elsewhere it is a
    no-op that still yields a usable LayerTimeline."""
    try:
        from airllm.airllm_llama_mlx import TransformerBlock
        from airllm.persist.mlx_model_persister import MlxModelPersister
    except Exception:  # pragma: no cover - non-mac / mlx missing
        with capture_layer_timeline() as tl:
            assert isinstance(tl, LayerTimeline)
        return

    orig_load = MlxModelPersister.load_model
    orig_call = TransformerBlock.__call__
    with capture_layer_timeline() as tl:
        assert isinstance(tl, LayerTimeline)
        assert MlxModelPersister.load_model is not orig_load
        assert TransformerBlock.__call__ is not orig_call
    assert MlxModelPersister.load_model is orig_load
    assert TransformerBlock.__call__ is orig_call
