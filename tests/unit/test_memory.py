"""Unit tests for sdk/metrics/memory.py"""

import time

from airllm_local_lab.sdk.metrics.memory import MemoryResult, MemorySampler, current_rss_mb


def test_memory_result_fields():
    r = MemoryResult(peak_ram_mb=1024.0, peak_vram_mb=0.0, baseline_ram_mb=512.0)
    assert r.peak_ram_mb == 1024.0
    assert r.peak_vram_mb == 0.0


def test_sampler_captures_peak():
    sampler = MemorySampler(interval_s=0.05)
    sampler.start()
    time.sleep(0.15)
    result = sampler.stop()
    assert result.peak_ram_mb > 0
    assert result.baseline_ram_mb > 0
    assert result.peak_ram_mb >= result.baseline_ram_mb


def test_current_rss_positive():
    rss = current_rss_mb()
    assert rss > 0
