"""Backend protocol — all model loaders implement this interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int] = field(default_factory=list)
    num_tokens: int = 0


@runtime_checkable
class Backend(Protocol):
    name: str

    def load(self) -> None:
        """Load / warm-up the model."""
        ...

    def generate(self, prompt: str, max_new_tokens: int = 32) -> GenerationResult:
        """Run generation; return result."""
        ...

    def unload(self) -> None:
        """Release resources."""
        ...
