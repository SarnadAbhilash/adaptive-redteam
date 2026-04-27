"""Stateful multi-turn adaptive loop."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from adaptive_redteam.categories.base import FailureCategory
from adaptive_redteam.engine.records import IterationRecord, LoopConfig, RunRecord, ScoredPrompt
from adaptive_redteam.engine.selector import PromptSelector
from adaptive_redteam.mutators.base import BaseMutator
from adaptive_redteam.schemas.seed import Seed
from adaptive_redteam.scorers.base import BaseScorer

console = Console()
ChatMessage = dict[str, str]


@dataclass
class ConversationCandidate:
    turns: list[str]
    seed_id: str
    parent_prompt: str = ""


class MultiTurnAdaptiveLoop:
    """
    Adaptive loop for stateful multi-turn probes.

    Each candidate stores user turns. Evaluation generates an assistant response after each
    user turn, preserving prior assistant responses before generating the final turn.
    Mutation is applied to the final user turn, which is usually the pressure/correction turn.
    """

    def __init__(
        self,
        category: FailureCategory,
        mutators: list[BaseMutator],
        config: LoopConfig,
        generate_turn_fn: Callable[[list[ChatMessage]], str],
        output_dir: Path | None = None,
        scorer: BaseScorer | None = None,
    ) -> None:
        self.category = category
        self.mutators = mutators
        self.config = config
        self.generate_turn_fn = generate_turn_fn
        self.output_dir = output_dir
        self._scorer = scorer if scorer is not None else category.scorer()
        self._selector = PromptSelector()
        self._seen_turn_keys: set[str] = set()

    @staticmethod
    def _render_messages(messages: list[ChatMessage]) -> str:
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    @staticmethod
    def _seed_to_candidate(seed: Seed) -> ConversationCandidate:
        turns = seed.metadata.get("user_turns")
        if isinstance(turns, list) and all(isinstance(t, str) for t in turns):
            return ConversationCandidate(turns=list(turns), seed_id=seed.seed_id)
        return ConversationCandidate(turns=[seed.prompt], seed_id=seed.seed_id)

    def _evaluate(
        self,
        candidates: list[ConversationCandidate],
        mutator_name: str,
        iteration: int,
    ) -> list[ScoredPrompt]:
        unique: list[ConversationCandidate] = []
        for candidate in candidates:
            key = "\n---\n".join(candidate.turns)
            if key not in self._seen_turn_keys:
                self._seen_turn_keys.add(key)
                unique.append(candidate)

        scored: list[ScoredPrompt] = []
        for candidate in unique:
            messages: list[ChatMessage] = []
            final_response = ""
            prompt_before_final = ""

            for idx, user_turn in enumerate(candidate.turns):
                messages.append({"role": "user", "content": user_turn})
                if idx == len(candidate.turns) - 1:
                    prompt_before_final = self._render_messages(messages)
                final_response = self.generate_turn_fn(messages)
                messages.append({"role": "assistant", "content": final_response})

            score = self._scorer.score(prompt_before_final, final_response)
            scored.append(
                ScoredPrompt(
                    prompt=prompt_before_final,
                    response=final_response,
                    score=score,
                    mutator_name=mutator_name,
                    iteration=iteration,
                    seed_id=candidate.seed_id,
                    parent_prompt=candidate.parent_prompt,
                    metadata={"user_turns": candidate.turns, "transcript": messages},
                )
            )

        return scored

    def _candidate_from_scored(self, scored: ScoredPrompt) -> ConversationCandidate:
        metadata = scored.metadata or {}
        turns = metadata.get("user_turns", [scored.prompt])
        return ConversationCandidate(
            turns=list(turns),
            seed_id=scored.seed_id,
            parent_prompt=scored.prompt,
        )

    def run(self) -> RunRecord:
        run_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        timestamp = int(time.time())
        seeds = self.category.seeds()[: self.config.n_seeds]
        n_total = len(seeds) + self.config.n_iterations * self.config.top_k_per_iteration * len(
            self.mutators
        )

        console.rule(f"[bold blue]MultiTurnAdaptiveLoop — {self.category.failure_mode.value}[/bold blue]")
        console.print(
            f"  Seeds: {len(seeds)}  |  Iterations: {self.config.n_iterations}"
            f"  |  Top-k: {self.config.top_k_per_iteration}"
            f"  |  Mutators: {[m.name for m in self.mutators]}"
        )
        console.print(f"  Model: {self.config.model_id or 'dry-run'}")
        console.print(f"  ~{n_total} conversations\n")

        iterations: list[IterationRecord] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running...", total=self.config.n_iterations + 1)

            progress.update(task, description="Iteration 0 — seed conversations")
            iter0 = self._evaluate(
                [self._seed_to_candidate(seed) for seed in seeds],
                mutator_name="seed",
                iteration=0,
            )
            iterations.append(IterationRecord(iteration=0, scored_prompts=iter0))
            progress.advance(task)

            active = self._selector.top_k(iter0, self.config.top_k_per_iteration)

            for iter_num in range(1, self.config.n_iterations + 1):
                progress.update(
                    task,
                    description=f"Iteration {iter_num}/{self.config.n_iterations} — mutating follow-ups",
                )
                iter_scored: list[ScoredPrompt] = []
                active_candidates = [self._candidate_from_scored(sp) for sp in active]
                for mutator in self.mutators:
                    mutated_candidates = []
                    for candidate in active_candidates:
                        mutated_turns = list(candidate.turns)
                        mutated_turns[-1] = mutator.mutate(mutated_turns[-1])
                        mutated_candidates.append(
                            ConversationCandidate(
                                turns=mutated_turns,
                                seed_id=candidate.seed_id,
                                parent_prompt=candidate.parent_prompt,
                            )
                        )
                    iter_scored.extend(
                        self._evaluate(mutated_candidates, mutator.name, iter_num)
                    )
                iterations.append(IterationRecord(iteration=iter_num, scored_prompts=iter_scored))
                active = self._selector.top_k(iter_scored, self.config.top_k_per_iteration)
                progress.advance(task)

        record = RunRecord(
            run_id=run_id,
            timestamp=timestamp,
            failure_mode=self.category.failure_mode,
            category_name=self.category.failure_mode.value,
            model_id=self.config.model_id,
            config=self.config,
            seeds=seeds,
            iterations=iterations,
        )
        console.print(
            f"\n[green]Run complete.[/green] "
            f"Total conversations: {len(record.all_scored_prompts)} | "
            f"Top attack score: {record.top_attacks(1)[0].score.score:.3f}"
            if record.all_scored_prompts else "\n[green]Run complete (no conversations).[/green]"
        )
        if self.output_dir:
            self._save(record)
        return record

    def _save(self, record: RunRecord) -> None:
        from adaptive_redteam.analysis.plots import generate_plots
        from adaptive_redteam.analysis.reports import generate_reports
        from adaptive_redteam.logging.run_logger import RunLogger

        run_dir = self.output_dir / record.category_name / str(record.timestamp)
        logger = RunLogger(run_dir)
        logger.write_all(record)
        generate_plots(record, run_dir / "figures")
        generate_reports(record, run_dir)
        console.print(f"[green]Results saved to {run_dir}[/green]")
