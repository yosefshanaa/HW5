"""Unit tests for sdk/viz/tables.py"""

from airllm_local_lab.sdk.viz.tables import economics_assumptions_table, hardware_table, precision_sweep_table


def test_precision_sweep_table_empty():
    result = precision_sweep_table([])
    assert "Precision" in result


def test_precision_sweep_table_row():
    rows = [
        {
            "precision": "fp16",
            "peak_ram_mb": 4096,
            "shard_size_gb": 14,
            "ttft_median_s": 2.5,
            "tpot_median_s": 0.5,
            "throughput_median_tps": 0.4,
            "quality_normalised": 0.8,
        }
    ]
    result = precision_sweep_table(rows)
    assert "fp16" in result
    assert "|" in result


def test_economics_assumptions_table():
    assumptions = [{"parameter": "CapEx", "value": "$1999", "source": "MacBook"}]
    result = economics_assumptions_table(assumptions)
    assert "CapEx" in result


def test_hardware_table():
    hw = {"CPU": {"spec": "M3 Pro", "implication": "No CUDA"}}
    result = hardware_table(hw)
    assert "M3 Pro" in result
    assert "CPU" in result
