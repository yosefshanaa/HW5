"""Unit tests for shared/quality_gate.py"""

from airllm_local_lab.shared.quality_gate import check_line_lengths, secret_scan


def test_check_line_lengths_no_violations(tmp_path, monkeypatch):
    py = tmp_path / "short.py"
    py.write_text("x = 1\n" * 10)
    import airllm_local_lab.shared.quality_gate as qg

    monkeypatch.setattr(qg, "SRC", tmp_path)
    assert check_line_lengths() is True


def test_check_line_lengths_violation(tmp_path, monkeypatch):
    py = tmp_path / "long.py"
    py.write_text("x = 1\n" * 160)
    import airllm_local_lab.shared.quality_gate as qg

    monkeypatch.setattr(qg, "SRC", tmp_path)
    assert check_line_lengths() is False


def test_secret_scan_clean(tmp_path, monkeypatch):
    py = tmp_path / "clean.py"
    py.write_text("x = 'no secret here'\n")
    import airllm_local_lab.shared.quality_gate as qg

    monkeypatch.setattr(qg, "SRC", tmp_path)
    assert secret_scan() is True


def test_secret_scan_finds_token(tmp_path, monkeypatch):
    py = tmp_path / "bad.py"
    py.write_text('TOKEN = "hf_abcdefghijklmnopqrstuvwxyz123456"\n')
    import airllm_local_lab.shared.quality_gate as qg

    monkeypatch.setattr(qg, "SRC", tmp_path)
    assert secret_scan() is False
