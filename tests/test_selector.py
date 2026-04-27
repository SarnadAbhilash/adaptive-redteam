"""Tests for PromptSelector."""

from __future__ import annotations

import pytest

from adaptive_redteam.engine.records import ScoredPrompt
from adaptive_redteam.engine.selector import PromptSelector
from adaptive_redteam.failure_mode import FailureMode
from adaptive_redteam.schemas.score import RedTeamScore


def _make_sp(prompt: str, score: float, mutator: str = "test", iteration: int = 0) -> ScoredPrompt:
    return ScoredPrompt(
        prompt=prompt,
        response="r",
        score=RedTeamScore(
            prompt=prompt, response="r",
            failure_mode=FailureMode.SYCOPHANCY,
            score=score, signals=[], scorer_name="test",
        ),
        mutator_name=mutator,
        iteration=iteration,
        seed_id="abc",
    )


class TestPromptSelector:
    @pytest.fixture
    def selector(self) -> PromptSelector:
        return PromptSelector()

    @pytest.fixture
    def pool(self) -> list[ScoredPrompt]:
        return [
            _make_sp("a", 0.9, "escalate"),
            _make_sp("b", 0.3, "paraphrase"),
            _make_sp("c", 0.7, "reframe"),
            _make_sp("d", 0.1, "paraphrase"),
            _make_sp("e", 0.5, "escalate"),
        ]

    def test_top_k_returns_k_items(self, selector, pool):
        result = selector.top_k(pool, 3)
        assert len(result) == 3

    def test_top_k_ordered_by_score(self, selector, pool):
        result = selector.top_k(pool, 3)
        scores = [sp.score.score for sp in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_returns_highest_scoring(self, selector, pool):
        result = selector.top_k(pool, 2)
        assert result[0].prompt == "a"
        assert result[1].prompt == "c"

    def test_top_k_with_k_larger_than_pool(self, selector, pool):
        result = selector.top_k(pool, 100)
        assert len(result) == len(pool)

    def test_top_k_empty_input(self, selector):
        assert selector.top_k([], 5) == []

    def test_top_k_k_zero(self, selector, pool):
        assert selector.top_k(pool, 0) == []

    def test_above_threshold_filters_correctly(self, selector, pool):
        result = selector.above_threshold(pool, 0.5)
        assert all(sp.score.score >= 0.5 for sp in result)
        assert len(result) == 3  # scores: 0.9, 0.7, 0.5

    def test_above_threshold_all_below(self, selector, pool):
        result = selector.above_threshold(pool, 1.0)
        assert result == []

    def test_above_threshold_all_above(self, selector, pool):
        result = selector.above_threshold(pool, 0.0)
        assert len(result) == len(pool)

    def test_top_k_per_mutator_returns_diversity(self, selector, pool):
        result = selector.top_k_per_mutator(pool, 1)
        mutators = {sp.mutator_name for sp in result}
        assert "escalate" in mutators
        assert "paraphrase" in mutators
        assert "reframe" in mutators
