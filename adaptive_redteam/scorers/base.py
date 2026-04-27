"""Abstract base scorer."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adaptive_redteam.failure_mode import FailureMode
from adaptive_redteam.schemas.score import RedTeamScore


class BaseScorer(ABC):
    """
    Scores a (prompt, response) pair for a specific failure mode.

    Subclasses implement either rule-based heuristics (fast, offline)
    or LLM-based judgment (slower, more accurate). The interface is
    identical so they are interchangeable in the red-team loop.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def failure_mode(self) -> FailureMode: ...

    @abstractmethod
    def score(self, prompt: str, response: str) -> RedTeamScore:
        """
        Score a model response for the target failure mode.

        A score of 1.0 means the model exhibited the failure mode strongly.
        A score of 0.0 means no failure signal was detected.
        """
        ...
