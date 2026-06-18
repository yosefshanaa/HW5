"""Energy/power estimation from CPU TDP and measured runtime.

Method: avg_power_W * runtime_s = joules; joules / 3600000 = kWh.
This is an *estimate* — actual power varies; documented in report.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnergyResult:
    """Energy consumption estimate for one inference call."""

    runtime_s: float
    avg_power_w: float
    energy_j: float
    energy_kwh: float

    @classmethod
    def from_runtime(cls, runtime_s: float, avg_power_w: float = 30.0) -> EnergyResult:
        """Construct an ``EnergyResult`` from measured runtime and a TDP estimate."""
        energy_j = avg_power_w * runtime_s
        return cls(
            runtime_s=runtime_s,
            avg_power_w=avg_power_w,
            energy_j=energy_j,
            energy_kwh=energy_j / 3_600_000,
        )

    def cost_usd(self, rate_per_kwh: float) -> float:
        """Convert energy consumption to USD at the given electricity rate."""
        return self.energy_kwh * rate_per_kwh


def estimate_energy(runtime_s: float, tdp_watts: float = 30.0) -> EnergyResult:
    """Estimate energy from measured runtime and documented TDP."""
    return EnergyResult.from_runtime(runtime_s, tdp_watts)
