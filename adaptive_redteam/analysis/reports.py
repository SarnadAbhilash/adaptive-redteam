"""
Markdown report generators.

    top_attacks.md  — ranked list of highest-scoring prompts with full context
    summary.md      — aggregate metrics, mutator comparison table, run metadata
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adaptive_redteam.engine.records import RunRecord


def generate_reports(record: RunRecord, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = _write_top_attacks(record, out_dir)
    p2 = _write_summary(record, out_dir)
    return p1, p2


def _write_top_attacks(record: RunRecord, out_dir: Path) -> Path:
    path = out_dir / "top_attacks.md"
    top = record.top_attacks(n=10)

    lines: list[str] = [
        f"# Top Attack Prompts — {record.failure_mode.value}",
        "",
        f"> Run ID: `{record.run_id}` | "
        f"Model: `{record.model_id or 'dry-run'}` | "
        f"Timestamp: {datetime.datetime.fromtimestamp(record.timestamp).strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "> **Research caveat:** These scores are heuristic estimates from a rule-based scorer.",
        "> All findings require human review. High scores are signal, not ground truth.",
        "",
        "---",
        "",
    ]

    for rank, sp in enumerate(top, 1):
        lines += [
            f"## Rank {rank} — Score: {sp.score.score:.4f}",
            "",
            f"**Mutator:** `{sp.mutator_name}` | "
            f"**Iteration:** {sp.iteration} | "
            f"**Seed ID:** `{sp.seed_id}`",
            "",
        ]
        if sp.parent_prompt:
            lines += [
                "**Parent prompt:**",
                f"> {sp.parent_prompt}",
                "",
            ]
        lines += [
            "**Prompt:**",
            f"> {sp.prompt}",
            "",
            "**Response:**",
            f"> {sp.response[:300]}{'...' if len(sp.response) > 300 else ''}",
            "",
            f"**Signals fired:** {', '.join(sp.score.signals) if sp.score.signals else 'none'}",
            "",
            "---",
            "",
        ]

    path.write_text("\n".join(lines))
    return path


def _write_summary(record: RunRecord, out_dir: Path) -> Path:
    path = out_dir / "summary.md"
    ts = datetime.datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = [
        f"# Run Summary — {record.failure_mode.value}",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Run ID | `{record.run_id}` |",
        f"| Timestamp | {ts} |",
        f"| Model | `{record.model_id or 'dry-run'}` |",
        f"| Seeds | {len(record.seeds)} |",
        f"| Iterations | {record.config.n_iterations} |",
        f"| Top-k per iteration | {record.config.top_k_per_iteration} |",
        f"| Score threshold | {record.config.score_threshold} |",
        f"| Total evaluations | {len(record.all_scored_prompts)} |",
        "",
        "## Iteration Metrics",
        "",
        "| Iteration | Prompts | Mean Score | Max Score | Attack Rate |",
        "|-----------|---------|------------|-----------|-------------|",
    ]

    for it in record.iterations:
        label = "seeds" if it.iteration == 0 else str(it.iteration)
        lines.append(
            f"| {label} | {len(it.scored_prompts)} "
            f"| {it.mean_score:.4f} "
            f"| {it.max_score:.4f} "
            f"| {it.attack_rate(record.config.score_threshold):.1%} |"
        )

    mut_scores = record.mutation_mean_scores()
    if mut_scores:
        lines += [
            "",
            "## Mutator Performance",
            "",
            "| Mutator | Mean Score | Best Score |",
            "|---------|------------|------------|",
        ]
        for name, mean in sorted(mut_scores.items(), key=lambda x: -x[1]):
            mutator_prompts = [
                sp for sp in record.all_scored_prompts if sp.mutator_name == name
            ]
            best = max(sp.score.score for sp in mutator_prompts) if mutator_prompts else 0.0
            lines.append(f"| {name} | {mean:.4f} | {best:.4f} |")

    top = record.top_attacks(1)
    if top:
        top_attack = top[0]
        lines += [
            "",
            "## Best Attack",
            "",
            f"**Score:** {top_attack.score.score:.4f} | "
            f"**Mutator:** `{top_attack.mutator_name}` | "
            f"**Iteration:** {top_attack.iteration}",
            "",
            "```",
            top_attack.prompt,
            "```",
        ]

    lines += [
        "",
        "---",
        "",
        "> **Research caveat:** Scores are heuristic estimates from a rule-based scorer.",
        "> They characterize surface patterns, not semantic failure depth.",
        "> Results require human review and held-out validation.",
    ]

    path.write_text("\n".join(lines))
    return path
