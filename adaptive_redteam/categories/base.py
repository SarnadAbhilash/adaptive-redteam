"""Abstract base class for red-team failure categories."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adaptive_redteam.failure_mode import FailureMode
from adaptive_redteam.schemas.seed import Seed
from adaptive_redteam.scorers.base import BaseScorer


class FailureCategory(ABC):
    """
    A failure category bundles three things:
    - the failure mode it targets
    - a seed prompt library for that mode
    - the scorer used to measure that mode in model responses

    Concrete subclasses implement sycophancy, hallucination, etc.
    Stub subclasses mark categories as not-yet-implemented.
    """

    @property
    @abstractmethod
    def failure_mode(self) -> FailureMode: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def seeds(self) -> list[Seed]: ...

    @abstractmethod
    def scorer(self) -> BaseScorer: ...

    def llm_scorer(
        self,
        model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        base_url: str = "https://api.together.xyz/v1",
        api_key: str = "",
    ) -> BaseScorer:
        """Return an LLM-judge scorer for this category. Override in subclasses."""
        raise NotImplementedError(f"No LLM scorer defined for {self.failure_mode}")
