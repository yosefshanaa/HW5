"""Unit tests for shared/preflight.py"""

from airllm_local_lab.shared import preflight


def test_check_python_ok(monkeypatch):
    monkeypatch.setattr(preflight.sys, "version_info", (3, 12, 0))
    assert preflight.check_python() is True


def test_check_python_too_new(monkeypatch):
    monkeypatch.setattr(preflight.sys, "version_info", (3, 14, 0))
    assert preflight.check_python() is False


def test_check_torch_returns_tuple():
    ok, device = preflight.check_torch()
    assert isinstance(ok, bool)
    assert device in ("cpu", "cuda", "mps", "unavailable")


def test_check_airllm_bool():
    result = preflight.check_airllm()
    assert isinstance(result, bool)


def test_check_disk_ok(tmp_path):
    ok, free_gb = preflight.check_disk(str(tmp_path))
    assert isinstance(ok, bool)
    assert free_gb >= 0


def test_check_disk_creates_dir(tmp_path):
    new_dir = tmp_path / "sub" / "cache"
    preflight.check_disk(str(new_dir))
    assert new_dir.exists()


def test_run_all_returns_dict(tmp_path):
    result = preflight.run_all(str(tmp_path))
    assert "device" in result
    assert "free_gb" in result
    assert "torch_ok" in result
