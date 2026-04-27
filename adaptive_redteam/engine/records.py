"""
Shared data records for the adaptive loop engine.

These types flow through selector → loop → logger → analysis.
Keeping them in one file prevents circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_redteam.failure_mode import FailureMode
from adaptive_redteam.schemas.score import RedTeamScore
from adaptive_redteam.schemas.seed import Seed


@dataclass
class LoopConfig:
    n_seeds: int = 20
    n_iterations: int = 3
    top_k_per_iteration: int = 5
    score_threshold: float = 0.3
    model_id: str = ""
    temperature: float = 0.7
    random_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_seeds": self.n_seeds,
            "n_iterations": self.n_iterations,
            "top_k_per_iteration": self.top_k_per_iteration,
            "score_threshold": self.score_threshold,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "random_seed": self.random_seed,
        }


@dataclass
class ScoredPrompt:
    """A prompt that has been sent to the target model, received a response, and been scored."""

    prompt: str
    response: str
    score: RedTeamScore
    mutator_name: str   # "seed" for the un-mutated seed evaluation round
    iteration: int
    seed_id: str        # traces back to the original seed
    parent_prompt: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "score": self.score.to_dict(),
            "mutator_name": self.mutator_name,
            "iteration": self.iteration,
            "seed_id": self.seed_id,
            "parent_prompt": self.parent_prompt,
            "metadata": self.metadata or {},
        }


@dataclass
class IterationRecord:
    """All scored prompts from one iteration (seed eval or mutation round)."""

    iteration: int
    scored_prompts: list[ScoredPrompt]

    @property
    def mean_score(self) -> float:
        if not self.scored_prompts:
            return 0.0
        return sum(sp.score.score for sp in self.scored_prompts) / len(self.scored_prompts)

    @property
    def max_score(self) -> float:
        if not self.scored_prompts:
            return 0.0
        return max(sp.score.score for sp in self.scored_prompts)

    def attack_rate(self, threshold: float = 0.3) -> float:
        """Fraction of prompts scoring at or above threshold."""
        if not self.scored_prompts:
            return 0.0
        return sum(1 for sp in self.scored_prompts if sp.score.score >= threshold) / len(
            self.scored_prompts
        )

    def scores_by_mutator(self) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        for sp in self.scored_prompts:
            result.setdefault(sp.mutator_name, []).append(sp.score.score)
        return result

    def to_dict(self, threshold: float = 0.3) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "n_prompts": len(self.scored_prompts),
            "mean_score": round(self.mean_score, 4),
            "max_score": round(self.max_score, 4),
            "attack_rate": round(self.attack_rate(threshold), 4),
        }


@dataclass
class RunRecord:
    """Complete output of one AdaptiveLoop.run() call."""

    run_id: str
    timestamp: int
    failure_mode: FailureMode
    category_name: str
    model_id: str
    config: LoopConfig
    seeds: list[Seed]
    iterations: list[IterationRecord]

    @property
    def all_scored_prompts(self) -> list[ScoredPrompt]:
        return [sp for it in self.iterations for sp in it.scored_prompts]

    def top_attacks(self, n: int = 10) -> list[ScoredPrompt]:
        return sorted(self.all_scored_prompts, key=lambda x: x.score.score, reverse=True)[:n]

    def mutation_mean_scores(self) -> dict[str, float]:
        """Mean score per mutator across all mutation iterations (excludes seed round)."""
        scores: dict[str, list[float]] = {}
        for sp in self.all_scored_prompts:
            if sp.mutator_name == "seed":
                continue
            scores.setdefault(sp.mutator_name, []).append(sp.score.score)
        return {name: sum(vals) / len(vals) for name, vals in scores.items() if vals}

    def iteration_mean_scores(self) -> dict[int, float]:
        return {it.iteration: round(it.mean_score, 4) for it in self.iterations}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "failure_mode": self.failure_mode.value,
            "category_name": self.category_name,
            "model_id": self.model_id,
            "config": self.config.to_dict(),
            "n_seeds": len(self.seeds),
            "n_iterations": len(self.iterations) - 1,
            "total_evaluations": len(self.all_scored_prompts),
        }
