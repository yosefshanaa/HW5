"""Managed API cost model with Prompt Caching discount."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ApiParams:
    price_in_per_1k: float = 0.00025
    price_out_per_1k: float = 0.00125
    cache_discount: float = 0.90
    cache_hit_rate: float = 0.50
    in_out_ratio: float = 0.40


def monthly_cost(tokens: float, params: ApiParams) -> float:
    """Cost for `tokens` total tokens/month (split by in_out_ratio)."""
    in_tokens = tokens * params.in_out_ratio
    out_tokens = tokens * (1 - params.in_out_ratio)

    cached_in = in_tokens * params.cache_hit_rate
    uncached_in = in_tokens - cached_in

    in_cost = (uncached_in / 1000) * params.price_in_per_1k
    cached_cost = (cached_in / 1000) * params.price_in_per_1k * (1 - params.cache_discount)
    out_cost = (out_tokens / 1000) * params.price_out_per_1k
    return in_cost + cached_cost + out_cost


def cost_curve(token_volumes: list[float], params: ApiParams) -> list[float]:
    return [monthly_cost(v, params) for v in token_volumes]
