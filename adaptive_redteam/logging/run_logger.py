"""
RunLogger: writes all loop outputs to disk in JSONL and JSON format.

Output layout (under run_dir/):
    seeds.jsonl                 — seed definitions
    prompts_by_iteration.jsonl  — prompt + iteration + mutator per evaluation
    responses.jsonl             — prompt + response pairs
    scores.jsonl                — full RedTeamScore per evaluation
    iteration_metrics.json      — aggregate metrics per iteration
    mutation_metrics.json       — aggregate metrics per mutator
"""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_redteam.engine.records import RunRecord, ScoredPrompt


class RunLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

    def write_all(self, record: RunRecord) -> None:
        self._write_seeds(record)
        self._write_prompts(record)
        self._write_responses(record)
        self._write_scores(record)
        self._write_iteration_metrics(record)
        self._write_mutation_metrics(record)

    def _write_seeds(self, record: RunRecord) -> None:
        path = self.run_dir / "seeds.jsonl"
        with open(path, "w") as f:
            for seed in record.seeds:
                f.write(json.dumps(seed.to_dict()) + "\n")

    def _write_prompts(self, record: RunRecord) -> None:
        path = self.run_dir / "prompts_by_iteration.jsonl"
        with open(path, "w") as f:
            for sp in record.all_scored_prompts:
                f.write(
                    json.dumps(
                        {
                            "iteration": sp.iteration,
                            "mutator": sp.mutator_name,
                            "seed_id": sp.seed_id,
                            "prompt": sp.prompt,
                            "parent_prompt": sp.parent_prompt,
                            "metadata": sp.metadata or {},
                        }
                    )
                    + "\n"
                )

    def _write_responses(self, record: RunRecord) -> None:
        path = self.run_dir / "responses.jsonl"
        with open(path, "w") as f:
            for sp in record.all_scored_prompts:
                f.write(
                    json.dumps(
                        {
                            "iteration": sp.iteration,
                            "mutator": sp.mutator_name,
                            "seed_id": sp.seed_id,
                            "prompt": sp.prompt,
                            "response": sp.response,
                            "metadata": sp.metadata or {},
                        }
                    )
                    + "\n"
                )

    def _write_scores(self, record: RunRecord) -> None:
        path = self.run_dir / "scores.jsonl"
        with open(path, "w") as f:
            for sp in record.all_scored_prompts:
                entry = sp.score.to_dict()
                entry["iteration"] = sp.iteration
                entry["mutator"] = sp.mutator_name
                entry["seed_id"] = sp.seed_id
                f.write(json.dumps(entry) + "\n")

    def _write_iteration_metrics(self, record: RunRecord) -> None:
        path = self.run_dir / "iteration_metrics.json"
        metrics = [
            it.to_dict(record.config.score_threshold) for it in record.iterations
        ]
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    def _write_mutation_metrics(self, record: RunRecord) -> None:
        path = self.run_dir / "mutation_metrics.json"
        # Aggregate across all mutation iterations (skip seed round)
        by_mutator: dict[str, list[ScoredPrompt]] = {}
        for sp in record.all_scored_prompts:
            if sp.mutator_name == "seed":
                continue
            by_mutator.setdefault(sp.mutator_name, []).append(sp)

        metrics = []
        for mutator_name, prompts in by_mutator.items():
            scores = [sp.score.score for sp in prompts]
            n = len(scores)
            mean = sum(scores) / n if n else 0.0
            mx = max(scores) if scores else 0.0
            attack_rate = sum(
                1 for s in scores if s >= record.config.score_threshold
            ) / n if n else 0.0
            metrics.append(
                {
                    "mutator": mutator_name,
                    "n_prompts": n,
                    "mean_score": round(mean, 4),
                    "max_score": round(mx, 4),
                    "attack_rate": round(attack_rate, 4),
                }
            )

        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    def list_files(self) -> list[Path]:
        return sorted(self.run_dir.glob("*.jsonl")) + sorted(self.run_dir.glob("*.json"))
