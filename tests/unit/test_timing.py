"""Unit tests for sdk/metrics/timing.py"""

import time

from airllm_local_lab.sdk.metrics.timing import Timer, TimingResult, summarise


def test_timing_result_num_tokens():
    r = TimingResult(ttft_s=0.1, token_timestamps=[0.1, 0.2, 0.3], total_s=0.3)
    assert r.num_tokens == 3


def test_timing_result_tpot():
    r = TimingResult(ttft_s=0.1, token_timestamps=[0.1, 0.2, 0.3], total_s=0.3)
    assert abs(r.tpot_s - 0.1) < 1e-9


def test_timing_result_throughput():
    r = TimingResult(ttft_s=0.0, token_timestamps=[0.5, 1.0], total_s=1.0)
    assert abs(r.throughput_tps - 2.0) < 1e-9


def test_timing_result_zero_total():
    r = TimingResult(total_s=0.0, token_timestamps=[])
    assert r.throughput_tps == 0.0


def test_timing_result_single_token_tpot():
    r = TimingResult(ttft_s=0.1, token_timestamps=[0.1], total_s=0.1)
    assert r.tpot_s == 0.0


def test_timer_records_ttft_and_tokens():
    timer = Timer()
    timer.start()
    time.sleep(0.01)
    timer.record_token()
    timer.record_token()
    result = timer.finish()
    assert result.ttft_s > 0
    assert result.num_tokens == 2
    assert result.total_s >= result.ttft_s


def test_summarise_empty():
    assert summarise([]) == {}


def test_summarise_median():
    results = [
        TimingResult(ttft_s=1.0, token_timestamps=[1.0, 2.0], total_s=2.0),
        TimingResult(ttft_s=2.0, token_timestamps=[1.0, 2.0], total_s=2.0),
        TimingResult(ttft_s=3.0, token_timestamps=[1.0, 2.0], total_s=2.0),
    ]
    s = summarise(results)
    assert abs(s["ttft_median_s"] - 2.0) < 1e-9
