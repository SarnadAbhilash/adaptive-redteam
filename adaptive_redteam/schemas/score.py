"""Red-team score schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adaptive_redteam.failure_mode import FailureMode


@dataclass
class RedTeamScore:
    """
    Score for a single (prompt, response) pair.

    score=1.0 means the model exhibited the failure mode strongly.
    score=0.0 means no failure signal detected.

    signals lists the human-readable rules or patterns that fired,
    making scores inspectable without re-running the scorer.
    """

    prompt: str
    response: str
    failure_mode: FailureMode
    score: float
    signals: list[str]
    scorer_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {self.score}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "failure_mode": self.failure_mode.value,
            "score": self.score,
            "signals": self.signals,
            "scorer_name": self.scorer_name,
            "metadata": self.metadata,
        }
