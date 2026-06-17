"""Unit tests for sdk/viz/plots.py — all run headless (Agg backend)."""

import pytest

from airllm_local_lab.sdk.viz import plots


@pytest.fixture(autouse=True)
def _tmp_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "ASSETS", tmp_path)
    return tmp_path


_ROWS = [
    {
        "precision": "fp16",
        "peak_ram_mb": 8192,
        "shard_size_gb": 14,
        "ttft_median_s": 5.0,
        "tpot_median_s": 1.0,
        "throughput_median_tps": 0.2,
        "quality_normalised": 0.8,
        "memory_fraction_of_fp16": 1.0,
        "throughput_iqr_tps": 0.05,
    },
    {
        "precision": "4bit",
        "peak_ram_mb": 2048,
        "shard_size_gb": 3,
        "ttft_median_s": 2.0,
        "tpot_median_s": 0.5,
        "throughput_median_tps": 0.5,
        "quality_normalised": 0.6,
        "memory_fraction_of_fp16": 0.25,
        "throughput_iqr_tps": 0.02,
    },
]


def test_f1_creates_file(tmp_path):
    p = plots.f1_memory_footprint(_ROWS)
    assert p.exists()
    assert p.suffix == ".png"


def test_f2_creates_file(tmp_path):
    p = plots.f2_latency(_ROWS)
    assert p.exists()


def test_f3_creates_file(tmp_path):
    p = plots.f3_throughput(_ROWS)
    assert p.exists()


def test_f4_creates_file(tmp_path):
    p = plots.f4_quality_vs_memory(_ROWS)
    assert p.exists()


def test_f5_creates_file_with_data(tmp_path):
    layer_dicts = [{"layer": i, "load_ms": 100.0, "compute_ms": 10.0} for i in range(5)]
    p = plots.f5_layer_timeline(layer_dicts)
    assert p.exists()


def test_f5_creates_file_empty(tmp_path):
    p = plots.f5_layer_timeline([])
    assert p.exists()


def test_f6_creates_file(tmp_path):
    p = plots.f6_page_cache_warmup([10.0], [8.0, 7.0, 7.5])
    assert p.exists()


def test_f7_creates_file(tmp_path):
    p = plots.f7_io_location({"internal_ssd": 12.5, "tmp": 6.0})
    assert p.exists()


def test_breakeven_chart_with_crossover(tmp_path):
    volumes = [0, 1_000_000, 5_000_000]
    curves = {"onprem": [50.0, 50.0, 50.0], "api": [0.0, 10.0, 50.0]}
    p = plots.breakeven_chart(volumes, curves, crossover=5_000_000)
    assert p.exists()


def test_breakeven_chart_no_crossover(tmp_path):
    volumes = [0, 1_000_000]
    curves = {"onprem": [100.0, 100.0], "api": [0.0, 10.0]}
    p = plots.breakeven_chart(volumes, curves, crossover=None)
    assert p.exists()
