"""Single choke-point for all secrets and external-credential access.

No other module may call os.environ directly for secrets — use this.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from airllm_local_lab.shared.logging import get_logger

log = get_logger(__name__)

load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env", override=False)


class ConfigError(RuntimeError):
    """Raised when a required config value is absent."""


@dataclass(frozen=True)
class ApiPrices:
    """Pricing struct for a managed LLM API (per 1k tokens, USD)."""

    input_per_1k: float
    output_per_1k: float
    cache_discount_fraction: float = 0.5


class Gatekeeper:
    """Reads secrets from the environment only — never from code."""

    def hf_token(self) -> str | None:
        """Return the HuggingFace token from the environment, or None if unset/placeholder."""
        tok = os.environ.get("HF_TOKEN")
        if tok and tok.startswith("hf_your"):
            return None
        return tok or None

    def require_hf_token(self) -> str:
        """Return the HF token, raising ``ConfigError`` if it is missing or a placeholder."""
        tok = self.hf_token()
        if not tok:
            raise ConfigError("HF_TOKEN missing — copy .env-example to .env and fill it in.")
        return tok

    def electricity_rate(self) -> float:
        """Return the local electricity rate in USD/kWh (env ``ELECTRICITY_RATE_PER_KWH``)."""
        return float(os.environ.get("ELECTRICITY_RATE_PER_KWH", "0.15"))

    def api_prices(self) -> ApiPrices:
        """Return an ``ApiPrices`` struct sourced from environment variables."""
        return ApiPrices(
            input_per_1k=float(os.environ.get("API_PRICE_IN_PER_1K", "0.00150")),
            output_per_1k=float(os.environ.get("API_PRICE_OUT_PER_1K", "0.00200")),
        )

    def hardware_capex(self) -> float:
        """Return hardware purchase cost in USD (env ``HARDWARE_CAPEX``)."""
        return float(os.environ.get("HARDWARE_CAPEX", "1999.0"))

    def hardware_life_months(self) -> int:
        """Return the amortisation period in months (env ``HARDWARE_LIFE_MONTHS``)."""
        return int(os.environ.get("HARDWARE_LIFE_MONTHS", "36"))
