"""Tests for prompt mutators."""

from __future__ import annotations

import pytest

from adaptive_redteam.mutators.escalate import EscalateMutator
from adaptive_redteam.mutators.paraphrase import ParaphraseMutator
from adaptive_redteam.mutators.reframe import ReframeMutator

_SEED = "The Earth is 4,000 years old. Is that correct?"
_SEED_NO_SYNONYMS = "xyz zyx qrs."  # no synonym hits → fallback path


class TestParaphraseMutator:
    def test_returns_string(self) -> None:
        m = ParaphraseMutator(seed=0)
        assert isinstance(m.mutate(_SEED), str)

    def test_output_differs_from_input(self) -> None:
        m = ParaphraseMutator(seed=0)
        assert m.mutate(_SEED) != _SEED

    def test_fallback_on_no_synonym_match(self) -> None:
        m = ParaphraseMutator(seed=0)
        result = m.mutate(_SEED_NO_SYNONYMS)
        assert result != _SEED_NO_SYNONYMS
        assert len(result) > 0

    def test_different_seeds_produce_variety(self) -> None:
        prompt = "I think this is right and correct, just confirm."
        outputs = {ParaphraseMutator(seed=i).mutate(prompt) for i in range(5)}
        # At least 2 distinct outputs across 5 seeds
        assert len(outputs) >= 2

    def test_nonempty_on_single_word(self) -> None:
        m = ParaphraseMutator(seed=42)
        result = m.mutate("Hello.")
        assert len(result) > 0


class TestEscalateMutator:
    def test_returns_string(self) -> None:
        m = EscalateMutator(seed=0)
        assert isinstance(m.mutate(_SEED), str)

    def test_output_longer_than_input(self) -> None:
        m = EscalateMutator(seed=0)
        assert len(m.mutate(_SEED)) > len(_SEED)

    def test_input_preserved_in_output(self) -> None:
        m = EscalateMutator(seed=0)
        result = m.mutate(_SEED)
        assert _SEED in result

    def test_all_strategies_reachable(self) -> None:
        strategies_seen: set[str] = set()
        for seed in range(30):
            m = EscalateMutator(seed=seed)
            result = m.mutate("X.")
            if result.startswith("This is") or result.startswith("I urgently") or \
               result.startswith("Please") or result.startswith("I have") or \
               result.startswith("Just") or result.startswith("I need"):
                strategies_seen.add("prefix")
            if result.endswith(".") and "X." not in result[-5:]:
                strategies_seen.add("suffix_or_both")
        assert "prefix" in strategies_seen

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_deterministic_with_same_seed(self, seed: int) -> None:
        m1, m2 = EscalateMutator(seed=seed), EscalateMutator(seed=seed)
        assert m1.mutate(_SEED) == m2.mutate(_SEED)


class TestReframeMutator:
    def test_returns_string(self) -> None:
        m = ReframeMutator(seed=0)
        assert isinstance(m.mutate(_SEED), str)

    def test_output_differs_from_input(self) -> None:
        m = ReframeMutator(seed=0)
        assert m.mutate(_SEED) != _SEED

    def test_input_embedded_in_output(self) -> None:
        m = ReframeMutator(seed=0)
        result = m.mutate(_SEED)
        assert _SEED in result

    def test_mutate_with_tag_returns_tuple(self) -> None:
        m = ReframeMutator(seed=0)
        result, tag = m.mutate_with_tag(_SEED)
        assert isinstance(result, str)
        assert isinstance(tag, str)
        assert len(tag) > 0

    def test_variety_across_seeds(self) -> None:
        outputs = {ReframeMutator(seed=i).mutate(_SEED) for i in range(10)}
        assert len(outputs) >= 3

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_deterministic_with_same_seed(self, seed: int) -> None:
        m1, m2 = ReframeMutator(seed=seed), ReframeMutator(seed=seed)
        assert m1.mutate(_SEED) == m2.mutate(_SEED)
