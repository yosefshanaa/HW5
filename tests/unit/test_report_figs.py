"""Unit tests for services/_report_figs.py — F5 regeneration + honest caption."""

from unittest.mock import patch

from airllm_local_lab.services import _report_figs


def test_render_f5_with_measured_data(tmp_path):
    (tmp_path / "layer_timeline.json").write_text(
        '[{"layer":0,"load_ms":80.0,"compute_ms":20.0,"total_ms":100.0},'
        '{"layer":1,"load_ms":80.0,"compute_ms":20.0,"total_ms":100.0}]'
    )
    with patch.object(_report_figs.plots, "f5_layer_timeline") as mock_plot:
        caption, blurb = _report_figs.render_f5(tmp_path)
    mock_plot.assert_called_once()
    assert len(mock_plot.call_args[0][0]) == 2  # regenerated from the 2 layers
    assert "Measured" in caption
    assert "80%" in caption
    assert "I/O-bound" in blurb
    assert "2 TinyLlama" in blurb


def test_render_f5_without_data_falls_back(tmp_path):
    with patch.object(_report_figs.plots, "f5_layer_timeline") as mock_plot:
        caption, blurb = _report_figs.render_f5(tmp_path)
    mock_plot.assert_called_once_with([])
    assert caption == "F5 — Per-layer load vs compute timeline"
    assert blurb == ""
