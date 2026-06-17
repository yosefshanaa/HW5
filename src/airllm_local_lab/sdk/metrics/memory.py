"""Peak RAM sampler using psutil; optional MPS memory tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class MemoryResult:
    peak_ram_mb: float = 0.0
    peak_vram_mb: float = 0.0
    baseline_ram_mb: float = 0.0


class MemorySampler:
    """Polls process RSS and (if MPS available) GPU memory at a given interval."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self._peak_ram: float = 0.0
        self._peak_vram: float = 0.0
        self._baseline: float = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

    def _sample_loop(self) -> None:
        import psutil

        proc = psutil.Process()
        while self._running:
            rss_mb = proc.memory_info().rss / 1e6
            if rss_mb > self._peak_ram:
                self._peak_ram = rss_mb
            try:
                import torch

                if torch.backends.mps.is_available():
                    vram_mb = torch.mps.current_allocated_memory() / 1e6
                    if vram_mb > self._peak_vram:
                        self._peak_vram = vram_mb
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self) -> None:
        import psutil

        self._baseline = psutil.Process().memory_info().rss / 1e6
        self._peak_ram = self._baseline
        self._peak_vram = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> MemoryResult:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        return MemoryResult(
            peak_ram_mb=self._peak_ram,
            peak_vram_mb=self._peak_vram,
            baseline_ram_mb=self._baseline,
        )


def current_rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1e6
