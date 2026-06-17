"""Unit tests for shared/version.py"""

from airllm_local_lab.shared.version import __version__


def test_version_string():
    assert __version__ == "1.00"


def test_version_is_string():
    assert isinstance(__version__, str)
