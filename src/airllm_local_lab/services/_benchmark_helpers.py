"""Pure helper functions for the benchmark pipeline — no I/O."""

from __future__ import annotations

import statistics

from airllm_local_lab.sdk.metrics.energy import estimate_energy
from airllm_local_lab.sdk.metrics.memory import MemorySampler
from airllm_local_lab.sdk.metrics.timing import Timer
from airllm_local_lab.sdk.model_loader.airllm_backend import AirLLMBackend
from airllm_local_lab.sdk.quality.rater import rate


def _run_single(backend: AirLLMBackend, prompt: str, max_new_tokens: int, cache_state: str) -> dict:
    timer = Timer()
    sampler = MemorySampler()
    sampler.start()
    timer.start()

    result = backend.generate(prompt, max_new_tokens=max_new_tokens)

    for _ in range(result.num_tokens):
        timer.record_token()
    timing = timer.finish()
    mem = sampler.stop()
    energy = estimate_energy(timing.total_s)
    quality = rate(result.text, prompt)

    return {
        "cache_state": cache_state,
        "ttft_s": round(timing.ttft_s, 4),
        "tpot_s": round(timing.tpot_s, 4),
        "throughput_tps": round(timing.throughput_tps, 4),
        "total_s": round(timing.total_s, 3),
        "peak_ram_mb": round(mem.peak_ram_mb, 1),
        "energy_j": round(energy.energy_j, 3),
        "quality_normalised": round(quality.normalised, 3),
        "output": result.text[:100],
    }


def _summarise_reps(rows: list[dict]) -> dict:
    def med(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return statistics.median(vals) if vals else 0.0

    def iqr(key: str) -> float:
        vals = sorted(r[key] for r in rows if isinstance(r.get(key), (int, float)))
        if len(vals) < 4:
            return 0.0
        q1 = vals[len(vals) // 4]
        q3 = vals[3 * len(vals) // 4]
        return q3 - q1

    return {
        "ttft_median_s": round(med("ttft_s"), 4),
        "tpot_median_s": round(med("tpot_s"), 4),
        "throughput_median_tps": round(med("throughput_tps"), 4),
        "throughput_iqr_tps": round(iqr("throughput_tps"), 4),
        "peak_ram_mb": round(med("peak_ram_mb"), 1),
        "energy_j_median": round(med("energy_j"), 3),
        "quality_normalised": round(med("quality_normalised"), 3),
        "reps": len(rows),
    }
