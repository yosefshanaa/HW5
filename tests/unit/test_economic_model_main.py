"""Unit tests for services/economic_model.py main() — mocked external I/O."""

import json
from unittest.mock import patch

from airllm_local_lab.services.economic_model import main


def test_main_creates_outputs(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    (tmp_path / "assets").mkdir()

    import airllm_local_lab.services.economic_model as em

    monkeypatch.setattr(em, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(em, "ASSETS", tmp_path / "assets")

    with (
        patch("airllm_local_lab.services.economic_model.breakeven_chart"),
        patch("airllm_local_lab.services.economic_model.economics_assumptions_table", return_value="| A | B |\n"),
    ):
        main()

    assert (tmp_path / "results" / "economics.json").exists()
    data = json.loads((tmp_path / "results" / "economics.json").read_text())
    assert "crossover_tokens" in data
    assert "assumptions" in data
