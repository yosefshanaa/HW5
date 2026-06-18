"""Unit tests for the data-driven branches of the report section builders."""

from airllm_local_lab.services._report_extras import section_raw_benchmark
from airllm_local_lab.services._report_theory import section_theory_iso


def test_section_theory_iso_with_e1_and_e3_data():
    """Exercise the E1 (I/O location) and E3 (page-cache) measured-data branches."""
    e1 = {"internal_ssd": 33.793, "tmp": 42.598}
    e3 = {"cold": [39.297], "warm": [30.014, 27.145, 27.096, 29.385], "speedup": 1.34}
    md = section_theory_iso("![F7](assets/F7.png)", "![F6](assets/F6.png)", e1_data=e1, e3_data=e3)

    # E1 branch
    assert "Measured results:" in md
    assert "33.793 s" in md
    assert "faster" in md
    # E3 branch (cold + each warm row + speedup line)
    assert "39.297" in md
    assert "30.014" in md
    assert "Cold→warm speedup" in md
    # Static sections still present
    assert "## 9. Theory Linkage" in md
    assert "ISO/IEC 25010" in md


def test_section_theory_iso_without_optional_data():
    md = section_theory_iso("F7", "F6")
    assert "Theory Linkage" in md
    assert "Measured results:" not in md


def test_section_raw_benchmark_empty_returns_blank():
    assert section_raw_benchmark([]) == ""


def test_section_raw_benchmark_with_rows_computes_stats():
    rows = [
        {
            "rep": 1,
            "cache_state": "cold",
            "ttft_s": 27.099,
            "throughput_tps": 0.7749,
            "peak_ram_mb": 1101.1,
            "energy_j": 813.0,
            "quality_normalised": 0.778,
        },
        {
            "rep": 2,
            "cache_state": "warm",
            "ttft_s": 27.959,
            "throughput_tps": 0.7511,
            "peak_ram_mb": 1108.5,
            "energy_j": 838.8,
            "quality_normalised": 0.778,
        },
        {
            "rep": 3,
            "cache_state": "warm",
            "ttft_s": 28.370,
            "throughput_tps": 0.7402,
            "peak_ram_mb": 1115.7,
            "energy_j": 851.1,
            "quality_normalised": 0.778,
        },
    ]
    md = section_raw_benchmark(rows)
    assert "Raw Benchmark Data" in md
    assert "Statistical summary" in md
    assert "median" in md
    assert "Cold vs warm" in md
    # median TTFT of the three reps
    assert "27.959" in md
