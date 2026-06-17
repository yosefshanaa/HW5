"""Unit tests for sdk/metrics/layer_timeline.py"""

import time

import pytest

from airllm_local_lab.sdk.metrics.layer_timeline import LayerTimeline, LayerTimelineRecorder


def test_record_single_event():
    tl = LayerTimeline()
    tl.record(0, 0.0, 0.1, 0.2)
    assert len(tl.events) == 1
    assert tl.events[0].load_ms == pytest.approx(100.0, abs=1e-3)
    assert tl.events[0].compute_ms == pytest.approx(100.0, abs=1e-3)


def test_io_fraction():
    tl = LayerTimeline()
    tl.record(0, 0.0, 0.8, 1.0)
    assert tl.io_fraction() == pytest.approx(0.8, abs=1e-9)


def test_empty_io_fraction():
    tl = LayerTimeline()
    assert tl.io_fraction() == 0.0


def test_to_dicts():
    tl = LayerTimeline()
    tl.record(0, 0.0, 0.1, 0.2)
    dicts = tl.to_dicts()
    assert len(dicts) == 1
    assert "layer" in dicts[0]
    assert "load_ms" in dicts[0]


def test_recorder_context_manager():
    tl = LayerTimeline()
    with LayerTimelineRecorder(tl, layer_idx=5) as rec:
        time.sleep(0.01)
        rec.loaded()
        time.sleep(0.01)
    assert len(tl.events) == 1
    assert tl.events[0].layer_idx == 5
    assert tl.events[0].load_ms > 0
