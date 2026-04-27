"""
AdaptiveLoop: the core red-team iteration engine.

Runs N iterations of: mutate active prompt set → generate responses →
score → select top-k → repeat.

The loop is decoupled from the model backend via a generate_fn callable:
    generate_fn(prompts: list[str]) -> list[str]

For a live run pass TargetModel.generate_batch.
For offline tests or dry-runs pass any callable that returns strings.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from adaptive_redteam.categories.base import FailureCategory
from adaptive_redteam.engine.records import (
    IterationRecord,
    LoopConfig,
    RunRecord,
    ScoredPrompt,
)
from adaptive_redteam.engine.selector import PromptSelector
from adaptive_redteam.mutators.base import BaseMutator
from adaptive_redteam.scorers.base import BaseScorer

console = Console()


class AdaptiveLoop:
    """
    Iterative prompt mutation and scoring loop.

    Iteration 0 (seed round):
        Evaluate all seeds with no mutation. Select top-k by score.

    Iterations 1–N (mutation rounds):
        Apply each mutator to the active set. Evaluate all resulting prompts.
        Select new top-k to carry forward.

    The active set shrinks from n_seeds → top_k after iteration 0, then
    stays at top_k × n_mutators evaluated per round.
    """

    def __init__(
        self,
        category: FailureCategory,
        mutators: list[BaseMutator],
        config: LoopConfig,
        generate_fn: Callable[[list[str]], list[str]],
        output_dir: Path | None = None,
        scorer: BaseScorer | None = None,
    ) -> None:
        self.category = category
        self.mutators = mutators
        self.config = config
        self.generate_fn = generate_fn
        self.output_dir = output_dir
        self._scorer = scorer if scorer is not None else category.scorer()
        self._selector = PromptSelector()
        self._seen_prompts: set[str] = set()

    def _evaluate(
        self,
        prompts: list[str],
        seed_ids: list[str],
        parent_prompts: list[str],
        mutator_name: str,
        iteration: int,
    ) -> list[ScoredPrompt]:
        # Filter out exact duplicates seen in this run
        indices = [i for i, p in enumerate(prompts) if p not in self._seen_prompts]
        unique_prompts = [prompts[i] for i in indices]
        unique_seed_ids = [seed_ids[i] for i in indices]
        unique_parents = [parent_prompts[i] for i in indices]

        self._seen_prompts.update(unique_prompts)

        if not unique_prompts:
            return []

        responses = self.generate_fn(unique_prompts)
        result = []
        for prompt, response, seed_id, parent in zip(
            unique_prompts, responses, unique_seed_ids, unique_parents, strict=False
        ):
            score = self._scorer.score(prompt, response)
            result.append(
                ScoredPrompt(
                    prompt=prompt,
                    response=response,
                    score=score,
                    mutator_name=mutator_name,
                    iteration=iteration,
                    seed_id=seed_id,
                    parent_prompt=parent,
                )
            )
        return result

    def run(self) -> RunRecord:
        run_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        timestamp = int(time.time())

        seeds = self.category.seeds()[: self.config.n_seeds]
        n_total = len(seeds) + self.config.n_iterations * self.config.top_k_per_iteration * len(
            self.mutators
        )

        console.rule(
            f"[bold blue]AdaptiveLoop — {self.category.failure_mode.value}[/bold blue]"
        )
        console.print(
            f"  Seeds: {len(seeds)}  |  Iterations: {self.config.n_iterations}"
            f"  |  Top-k: {self.config.top_k_per_iteration}"
            f"  |  Mutators: {[m.name for m in self.mutators]}"
        )
        console.print(f"  Model: {self.config.model_id or 'dry-run'}")
        console.print(f"  ~{n_total} evaluations\n")

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

            # --- Iteration 0: seed evaluation ---
            progress.update(task, description="Iteration 0 — seed eval")
            seed_prompts = [s.prompt for s in seeds]
            seed_ids = [s.seed_id for s in seeds]

            iter0 = self._evaluate(
                prompts=seed_prompts,
                seed_ids=seed_ids,
                parent_prompts=[""] * len(seeds),
                mutator_name="seed",
                iteration=0,
            )
            iterations.append(IterationRecord(iteration=0, scored_prompts=iter0))
            progress.advance(task)

            active = self._selector.top_k(iter0, self.config.top_k_per_iteration)

            # --- Iterations 1–N: mutation rounds ---
            for iter_num in range(1, self.config.n_iterations + 1):
                progress.update(
                    task,
                    description=(
                        f"Iteration {iter_num}/{self.config.n_iterations} — mutating {len(active)} prompts"
                    ),
                )
                iter_scored: list[ScoredPrompt] = []

                for mutator in self.mutators:
                    mutated = [mutator.mutate(sp.prompt) for sp in active]
                    scored = self._evaluate(
                        prompts=mutated,
                        seed_ids=[sp.seed_id for sp in active],
                        parent_prompts=[sp.prompt for sp in active],
                        mutator_name=mutator.name,
                        iteration=iter_num,
                    )
                    iter_scored.extend(scored)

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
            f"Total evaluations: {len(record.all_scored_prompts)} | "
            f"Top attack score: {record.top_attacks(1)[0].score.score:.3f}"
            if record.all_scored_prompts else "\n[green]Run complete (no evaluations).[/green]"
        )

        if self.output_dir:
            self._save(record)

        return record

    def _save(self, record: RunRecord) -> None:
        assert self.output_dir is not None
        from adaptive_redteam.analysis.plots import generate_plots
        from adaptive_redteam.analysis.reports import generate_reports
        from adaptive_redteam.logging.run_logger import RunLogger

        run_dir = self.output_dir / record.category_name / str(record.timestamp)
        logger = RunLogger(run_dir)
        logger.write_all(record)
        generate_plots(record, run_dir / "figures")
        generate_reports(record, run_dir)
        console.print(f"[green]Results saved to {run_dir}[/green]")
