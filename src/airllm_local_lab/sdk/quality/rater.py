"""Deterministic rubric-based output quality rater.

Rubric (0–3 per criterion, total 0–9 → normalised 0–1):
  coherence     — grammatical, non-repetitive, not gibberish
  correctness   — factual / on-topic for the prompt
  completeness  — addresses the full prompt, not truncated mid-sentence
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QualityScore:
    coherence: int = 0
    correctness: int = 0
    completeness: int = 0

    @property
    def total(self) -> int:
        return self.coherence + self.correctness + self.completeness

    @property
    def normalised(self) -> float:
        return self.total / 9.0

    def to_dict(self) -> dict:
        return {
            "coherence": self.coherence,
            "correctness": self.correctness,
            "completeness": self.completeness,
            "total": self.total,
            "normalised": round(self.normalised, 3),
        }


_GIBBERISH_PATTERN = re.compile(r"([^\w\s,\.\!\?]){5,}")
_REPEAT_PATTERN = re.compile(r"(\b\w+\b)(?:\s+\1){4,}")


def _coherence_score(text: str) -> int:
    if not text or len(text.split()) < 2:
        return 0
    if _GIBBERISH_PATTERN.search(text):
        return 1
    if _REPEAT_PATTERN.search(text.lower()):
        return 1
    if len(text.split()) >= 6:
        return 3
    return 2


def _correctness_score(text: str, prompt: str) -> int:
    if not text:
        return 0
    prompt_words = set(re.findall(r"\b\w{4,}\b", prompt.lower()))
    output_words = set(re.findall(r"\b\w{4,}\b", text.lower()))
    overlap = len(prompt_words & output_words) / max(len(prompt_words), 1)
    if overlap >= 0.5:
        return 3
    if overlap >= 0.2:
        return 2
    return 1


def _completeness_score(text: str) -> int:
    if not text:
        return 0
    stripped = text.strip()
    ends_properly = stripped[-1] in ".!?:\"'" if stripped else False
    word_count = len(stripped.split())
    if word_count >= 20 and ends_properly:
        return 3
    if word_count >= 8:
        return 2
    return 1


def rate(output: str, prompt: str = "") -> QualityScore:
    return QualityScore(
        coherence=_coherence_score(output),
        correctness=_correctness_score(output, prompt),
        completeness=_completeness_score(output),
    )
