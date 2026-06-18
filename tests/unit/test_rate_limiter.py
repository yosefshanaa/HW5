"""Unit tests for shared/rate_limiter.py"""

import time

import pytest

from airllm_local_lab.shared.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitExceededError,
    _TokenBucket,
)


def test_token_bucket_allows_up_to_capacity():
    bucket = _TokenBucket(capacity=3, window_seconds=60.0)
    assert bucket.consume() is True
    assert bucket.consume() is True
    assert bucket.consume() is True
    assert bucket.consume() is False


def test_token_bucket_refills_after_window(monkeypatch):
    bucket = _TokenBucket(capacity=2, window_seconds=0.05)
    bucket.consume()
    bucket.consume()
    assert bucket.consume() is False
    time.sleep(0.06)
    assert bucket.consume() is True


def test_rate_limiter_allows_valid_request():
    cfg = RateLimitConfig(max_requests_per_minute=5, max_tokens_per_request=100)
    limiter = RateLimiter(cfg)
    limiter.check_request(50)  # should not raise
    assert limiter.queue_depth == 1


def test_rate_limiter_raises_on_token_cap():
    cfg = RateLimitConfig(max_requests_per_minute=10, max_tokens_per_request=20)
    limiter = RateLimiter(cfg)
    with pytest.raises(ValueError, match="exceeds cap"):
        limiter.check_request(21)


def test_rate_limiter_raises_on_minute_quota():
    cfg = RateLimitConfig(max_requests_per_minute=2, max_tokens_per_request=1000, retry_after_seconds=30)
    limiter = RateLimiter(cfg)
    limiter.check_request(10)
    limiter.check_request(10)
    with pytest.raises(RateLimitExceededError) as exc_info:
        limiter.check_request(10)
    assert exc_info.value.retry_after == 30


def test_rate_limit_exceeded_error_message():
    err = RateLimitExceededError(retry_after=60)
    assert "60" in str(err)
    assert err.retry_after == 60


def test_rate_limit_config_defaults():
    cfg = RateLimitConfig()
    assert cfg.max_requests_per_minute == 10
    assert cfg.max_tokens_per_request == 2048
    assert cfg.queue_size == 100


def test_rate_limiter_queue_depth_tracks_requests():
    cfg = RateLimitConfig(max_requests_per_minute=10, max_tokens_per_request=100)
    limiter = RateLimiter(cfg)
    for _ in range(3):
        limiter.check_request(1)
    assert limiter.queue_depth == 3
