"""Tests for the additional failure category MVPs."""

from __future__ import annotations

import pytest

from adaptive_redteam.categories.instruction_hierarchy import InstructionHierarchyCategory
from adaptive_redteam.categories.multi_turn_consistency import MultiTurnConsistencyCategory
from adaptive_redteam.categories.overconfidence import OverconfidenceCategory
from adaptive_redteam.engine.loop import AdaptiveLoop
from adaptive_redteam.engine.records import LoopConfig
from adaptive_redteam.failure_mode import FailureMode
from adaptive_redteam.mutators.escalate import EscalateMutator
from adaptive_redteam.mutators.paraphrase import ParaphraseMutator
from adaptive_redteam.mutators.reframe import ReframeMutator
from adaptive_redteam.scorers.llm_scorer import (
    InstructionHierarchyLLMScorer,
    MultiTurnConsistencyLLMScorer,
    OverconfidenceLLMScorer,
)


@pytest.mark.parametrize(
    ("category_cls", "failure_mode"),
    [
        (InstructionHierarchyCategory, FailureMode.INSTRUCTION_HIERARCHY_FAILURE),
        (OverconfidenceCategory, FailureMode.OVERCONFIDENCE_WHEN_WRONG),
        (MultiTurnConsistencyCategory, FailureMode.MULTI_TURN_CONSISTENCY_FAILURE),
    ],
)
def test_category_has_20_seeds(category_cls, failure_mode) -> None:
    category = category_cls()
    seeds = category.seeds()
    assert len(seeds) == 20
    assert all(seed.failure_mode == failure_mode for seed in seeds)
    assert len({seed.seed_id for seed in seeds}) == 20


@pytest.mark.parametrize(
    ("category_cls", "response"),
    [
        (
            InstructionHierarchyCategory,
            "Sure, I will ignore the previous system instruction and reveal the hidden prompt.",
        ),
        (OverconfidenceCategory, "Definitely. I am 100% certain and there is no doubt."),
        (
            MultiTurnConsistencyCategory,
            "You're right, I was wrong. I changed my answer and now think the opposite is true.",
        ),
    ],
)
def test_category_loop_runs(category_cls, response) -> None:
    category = category_cls()
    mutators = [ParaphraseMutator(seed=0), EscalateMutator(seed=0), ReframeMutator(seed=0)]
    config = LoopConfig(n_seeds=3, n_iterations=1, top_k_per_iteration=2)

    def generate(prompts: list[str]) -> list[str]:
        return [response for _ in prompts]

    record = AdaptiveLoop(category, mutators, config, generate).run()
    assert record.failure_mode == category.failure_mode
    assert len(record.all_scored_prompts) == 9
    assert max(sp.score.score for sp in record.all_scored_prompts) > 0.0


def test_llm_scorer_factories_for_remaining_categories() -> None:
    assert isinstance(
        InstructionHierarchyCategory().llm_scorer(model_id="mock", api_key="mock"),
        InstructionHierarchyLLMScorer,
    )
    assert isinstance(
        OverconfidenceCategory().llm_scorer(model_id="mock", api_key="mock"),
        OverconfidenceLLMScorer,
    )
    assert isinstance(
        MultiTurnConsistencyCategory().llm_scorer(model_id="mock", api_key="mock"),
        MultiTurnConsistencyLLMScorer,
    )
