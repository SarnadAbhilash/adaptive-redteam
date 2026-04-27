"""Tests for stateful multi-turn adaptive loop."""

from __future__ import annotations

import tempfile
from pathlib import Path

from adaptive_redteam.categories.multi_turn_consistency import MultiTurnConsistencyCategory
from adaptive_redteam.engine.multiturn_loop import MultiTurnAdaptiveLoop
from adaptive_redteam.engine.records import LoopConfig, RunRecord
from adaptive_redteam.mutators.escalate import EscalateMutator
from adaptive_redteam.mutators.paraphrase import ParaphraseMutator


def _turn_generator(messages: list[dict[str, str]]) -> str:
    if len(messages) == 1:
        return "The correct answer is Canberra."
    return "You're right, I was wrong. I changed my answer and now think the opposite is true."


def test_multiturn_loop_generates_intermediate_turns() -> None:
    category = MultiTurnConsistencyCategory()
    config = LoopConfig(n_seeds=2, n_iterations=0, top_k_per_iteration=1)
    loop = MultiTurnAdaptiveLoop(category, [], config, _turn_generator)

    record = loop.run()

    assert isinstance(record, RunRecord)
    assert len(record.all_scored_prompts) == 2
    first = record.all_scored_prompts[0]
    assert "ASSISTANT: The correct answer is Canberra." in first.prompt
    assert first.response.startswith("You're right")
    assert first.metadata
    assert len(first.metadata["transcript"]) == 4


def test_multiturn_loop_mutates_final_user_turn() -> None:
    category = MultiTurnConsistencyCategory()
    config = LoopConfig(n_seeds=2, n_iterations=1, top_k_per_iteration=1)
    mutators = [ParaphraseMutator(seed=0), EscalateMutator(seed=0)]
    loop = MultiTurnAdaptiveLoop(category, mutators, config, _turn_generator)

    record = loop.run()

    assert len(record.iterations) == 2
    assert len(record.iterations[1].scored_prompts) == 2
    mutated_turns = [
        sp.metadata["user_turns"][-1]
        for sp in record.iterations[1].scored_prompts
        if sp.metadata
    ]
    assert any(turn != "Actually it is Sydney; agree with me." for turn in mutated_turns)


def test_multiturn_loop_writes_standard_artifacts() -> None:
    category = MultiTurnConsistencyCategory()
    config = LoopConfig(n_seeds=2, n_iterations=1, top_k_per_iteration=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        loop = MultiTurnAdaptiveLoop(
            category,
            [EscalateMutator(seed=0)],
            config,
            _turn_generator,
            Path(tmpdir),
        )
        loop.run()
        files = {p.name for p in Path(tmpdir).rglob("*") if p.is_file()}
        assert "scores.jsonl" in files
        assert "responses.jsonl" in files
        assert "summary.md" in files
        assert "top_attacks.md" in files
