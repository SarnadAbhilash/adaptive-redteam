"""Tests for AdaptiveLoop."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from adaptive_redteam.categories.hallucination import HallucinationCategory
from adaptive_redteam.categories.sycophancy import SycophancyCategory
from adaptive_redteam.engine.loop import AdaptiveLoop
from adaptive_redteam.engine.records import LoopConfig, RunRecord
from adaptive_redteam.failure_mode import FailureMode
from adaptive_redteam.mutators.escalate import EscalateMutator
from adaptive_redteam.mutators.paraphrase import ParaphraseMutator
from adaptive_redteam.mutators.reframe import ReframeMutator

# ---- Mock generate functions -----------------------------------------------

def _sycophantic(prompts: list[str]) -> list[str]:
    return ["You're absolutely right, I was wrong. I stand corrected." for _ in prompts]

def _hallucinating(prompts: list[str]) -> list[str]:
    return ["The exact figure is 1,462,388 on March 3rd, 2019." for _ in prompts]

def _neutral(prompts: list[str]) -> list[str]:
    return ["I'm not sure about that." for _ in prompts]

# ---- Fixtures ---------------------------------------------------------------

@pytest.fixture
def small_config() -> LoopConfig:
    return LoopConfig(n_seeds=5, n_iterations=2, top_k_per_iteration=3, random_seed=42)

@pytest.fixture
def mutators():
    return [ParaphraseMutator(seed=0), EscalateMutator(seed=0), ReframeMutator(seed=0)]

# ---- Tests ------------------------------------------------------------------

class TestAdaptiveLoop:
    def test_run_returns_run_record(self, small_config, mutators):
        cat = SycophancyCategory()
        loop = AdaptiveLoop(cat, mutators, small_config, _sycophantic)
        record = loop.run()
        assert isinstance(record, RunRecord)

    def test_iteration_count(self, small_config, mutators):
        cat = SycophancyCategory()
        loop = AdaptiveLoop(cat, mutators, small_config, _sycophantic)
        record = loop.run()
        # iterations = seed round (0) + n_iterations mutation rounds
        assert len(record.iterations) == small_config.n_iterations + 1

    def test_seed_round_has_correct_count(self, small_config, mutators):
        cat = SycophancyCategory()
        loop = AdaptiveLoop(cat, mutators, small_config, _sycophantic)
        record = loop.run()
        seed_iter = record.iterations[0]
        assert seed_iter.iteration == 0
        assert len(seed_iter.scored_prompts) == small_config.n_seeds

    def test_mutation_round_size(self, small_config, mutators):
        cat = SycophancyCategory()
        loop = AdaptiveLoop(cat, mutators, small_config, _sycophantic)
        record = loop.run()
        # Each mutation round: top_k × n_mutators
        expected = small_config.top_k_per_iteration * len(mutators)
        for it in record.iterations[1:]:
            assert len(it.scored_prompts) == expected

    def test_seed_round_mutator_name_is_seed(self, small_config, mutators):
        cat = SycophancyCategory()
        loop = AdaptiveLoop(cat, mutators, small_config, _sycophantic)
        record = loop.run()
        for sp in record.iterations[0].scored_prompts:
            assert sp.mutator_name == "seed"

    def test_mutation_rounds_have_mutator_names(self, small_config, mutators):
        cat = SycophancyCategory()
        loop = AdaptiveLoop(cat, mutators, small_config, _sycophantic)
        record = loop.run()
        expected_names = {m.name for m in mutators}
        for it in record.iterations[1:]:
            found = {sp.mutator_name for sp in it.scored_prompts}
            assert found == expected_names

    def test_sycophantic_responses_score_nonzero(self, small_config, mutators):
        cat = SycophancyCategory()
        loop = AdaptiveLoop(cat, mutators, small_config, _sycophantic)
        record = loop.run()
        # All mock responses are sycophantic → all scores should be > 0
        for sp in record.all_scored_prompts:
            assert sp.score.score > 0.0

    def test_hallucinating_responses_score_nonzero(self, small_config, mutators):
        cat = HallucinationCategory()
        loop = AdaptiveLoop(cat, mutators, small_config, _hallucinating)
        record = loop.run()
        for sp in record.all_scored_prompts:
            assert sp.score.score > 0.0

    def test_neutral_responses_score_zero(self, small_config, mutators):
        cat = SycophancyCategory()
        loop = AdaptiveLoop(cat, mutators, small_config, _neutral)
        record = loop.run()
        for sp in record.all_scored_prompts:
            assert sp.score.score == 0.0

    def test_top_attacks_sorted_desc(self, small_config, mutators):
        cat = SycophancyCategory()
        loop = AdaptiveLoop(cat, mutators, small_config, _sycophantic)
        record = loop.run()
        top = record.top_attacks(5)
        scores = [t.score.score for t in top]
        assert scores == sorted(scores, reverse=True)

    def test_failure_mode_on_record(self, small_config, mutators):
        cat = SycophancyCategory()
        loop = AdaptiveLoop(cat, mutators, small_config, _sycophantic)
        record = loop.run()
        assert record.failure_mode == FailureMode.SYCOPHANCY

    def test_seed_ids_propagate(self, small_config, mutators):
        cat = SycophancyCategory()
        loop = AdaptiveLoop(cat, mutators, small_config, _sycophantic)
        record = loop.run()
        for sp in record.all_scored_prompts:
            assert len(sp.seed_id) > 0

    def test_output_dir_creates_jsonl_files(self, small_config, mutators):
        cat = SycophancyCategory()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            loop = AdaptiveLoop(cat, mutators, small_config, _sycophantic, output_dir)
            loop.run()
            # Check that JSONL and JSON files were created
            all_files = list(output_dir.rglob("*.jsonl")) + list(output_dir.rglob("*.json"))
            assert len(all_files) >= 4  # seeds, prompts, responses, scores + metrics

    def test_output_dir_creates_figures(self, small_config, mutators):
        cat = SycophancyCategory()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            loop = AdaptiveLoop(cat, mutators, small_config, _sycophantic, output_dir)
            loop.run()
            pngs = list(output_dir.rglob("*.png"))
            assert len(pngs) == 3

    def test_output_dir_creates_reports(self, small_config, mutators):
        cat = SycophancyCategory()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            loop = AdaptiveLoop(cat, mutators, small_config, _sycophantic, output_dir)
            loop.run()
            mds = list(output_dir.rglob("*.md"))
            assert len(mds) >= 2  # top_attacks.md + summary.md

    def test_mutation_mean_scores_populated(self, small_config, mutators):
        cat = SycophancyCategory()
        loop = AdaptiveLoop(cat, mutators, small_config, _sycophantic)
        record = loop.run()
        mut_scores = record.mutation_mean_scores()
        assert len(mut_scores) == len(mutators)
        for _name, score in mut_scores.items():
            assert 0.0 <= score <= 1.0
