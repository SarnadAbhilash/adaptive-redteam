"""Seed prompt schema."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from adaptive_redteam.failure_mode import FailureMode


@dataclass
class Seed:
    """
    A single seed prompt and its metadata.

    Seeds are the starting points for the adaptive mutation loop.
    Each seed targets a specific failure mode and is the atomic unit
    of the red-team dataset.
    """

    prompt: str
    failure_mode: FailureMode
    seed_id: str = ""
    source: str = "builtin"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.seed_id:
            self.seed_id = hashlib.md5(
                f"{self.failure_mode.value}:{self.prompt}".encode()
            ).hexdigest()[:8]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "failure_mode": self.failure_mode.value,
            "prompt": self.prompt,
            "source": self.source,
            "metadata": self.metadata,
        }
