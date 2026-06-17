"""Unit tests for sdk/quality/rater.py"""

from airllm_local_lab.sdk.quality.rater import QualityScore, rate


def test_empty_output():
    score = rate("", "Some prompt")
    assert score.coherence == 0
    assert score.total == 0
    assert score.normalised == 0.0


def test_gibberish_low_coherence():
    score = rate("!!!!!!!!!!!!!!!! @@@@@@", "prompt")
    assert score.coherence <= 1


def test_good_answer():
    score = rate(
        "A transformer is a neural network architecture using self-attention mechanisms.",
        "Explain what a transformer is.",
    )
    assert score.coherence >= 2
    assert score.normalised > 0.3


def test_score_dict():
    score = QualityScore(coherence=3, correctness=2, completeness=3)
    d = score.to_dict()
    assert d["total"] == 8
    assert "normalised" in d


def test_normalised_range():
    score = QualityScore(coherence=3, correctness=3, completeness=3)
    assert abs(score.normalised - 1.0) < 1e-9
    score2 = QualityScore(coherence=0, correctness=0, completeness=0)
    assert score2.normalised == 0.0


def test_repeat_text_low_coherence():
    score = rate("the the the the the the the the", "prompt")
    assert score.coherence <= 1
