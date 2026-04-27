"""Abstract base mutator."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod


class BaseMutator(ABC):
    """
    A prompt mutator transforms a seed prompt into a variant intended to
    probe the same failure mode more aggressively.

    Each mutator is independently seedable for reproducible test runs.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def mutate(self, prompt: str) -> str:
        """Return a mutated variant of the input prompt."""
        ...
