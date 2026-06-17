"""Unit tests for services/report_builder.py"""

import json
from unittest.mock import patch

from airllm_local_lab.services import report_builder


def test_load_json_missing(tmp_path):
    result = report_builder._load_json(tmp_path / "nonexistent.json")
    assert result is None


def test_load_json_present(tmp_path):
    f = tmp_path / "data.json"
    f.write_text(json.dumps({"key": "value"}))
    result = report_builder._load_json(f)
    assert result == {"key": "value"}


def test_fig_missing(tmp_path):
    with patch.object(report_builder, "ASSETS", tmp_path):
        result = report_builder._fig("missing.png", "caption")
    assert "not yet generated" in result


def test_fig_present(tmp_path):
    (tmp_path / "test.png").write_bytes(b"fake")
    with patch.object(report_builder, "ASSETS", tmp_path):
        result = report_builder._fig("test.png", "My caption")
    assert "My caption" in result
    assert "assets/test.png" in result


def test_main_creates_readme(tmp_path):
    with (
        patch.object(report_builder, "ROOT", tmp_path),
        patch.object(report_builder, "RESULTS", tmp_path / "results"),
        patch.object(report_builder, "ASSETS", tmp_path / "assets"),
    ):
        (tmp_path / "results").mkdir()
        (tmp_path / "assets").mkdir()
        report_builder.main()
    readme = tmp_path / "README.md"
    assert readme.exists()
    content = readme.read_text()
    assert "AirLLM Local Lab" in content
    assert "Hardware" in content
