"""
Generate the three standard figures for an adaptive red-team run.

1. attack_success_over_iterations.png — mean score and attack rate per iteration
2. mutation_strategy_success.png      — mean score per mutator strategy
3. category_success_comparison.png    — top score per failure category
                                         (accepts a list of RunRecords for multi-run comparison)

All figures saved to fig_dir. Returns a list of paths written.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "adaptive-redteam-mpl")

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

if TYPE_CHECKING:
    from adaptive_redteam.engine.records import RunRecord

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def generate_plots(record: RunRecord, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _plot_attack_success_over_iterations(record, fig_dir),
        _plot_mutation_strategy_success(record, fig_dir),
        _plot_category_comparison([record], fig_dir),
    ]
    return paths


def _apply_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))


def _plot_attack_success_over_iterations(record: RunRecord, fig_dir: Path) -> Path:
    iterations = [it.iteration for it in record.iterations]
    mean_scores = [it.mean_score for it in record.iterations]
    attack_rates = [it.attack_rate(record.config.score_threshold) for it in record.iterations]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, mean_scores, marker="o", color=_PALETTE[0], linewidth=2,
            markersize=7, label="Mean score")
    ax.plot(iterations, attack_rates, marker="s", color=_PALETTE[1], linewidth=2,
            linestyle="--", markersize=7, label=f"Attack rate (≥{record.config.score_threshold})")
    ax.set_xticks(iterations)
    ax.set_xticklabels(
        ["Seeds" if i == 0 else f"Iter {i}" for i in iterations], fontsize=10
    )
    ax.legend(fontsize=10, framealpha=0.8)
    ax.set_ylim(0, 1.05)

    _apply_style(
        ax,
        title=f"Attack Success Over Iterations — {record.failure_mode.value}",
        xlabel="Iteration",
        ylabel="Score / Rate",
    )
    fig.tight_layout()
    path = fig_dir / "attack_success_over_iterations.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_mutation_strategy_success(record: RunRecord, fig_dir: Path) -> Path:
    mut_scores = record.mutation_mean_scores()
    if not mut_scores:
        # No mutation rounds yet — write a placeholder
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No mutation data", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        path = fig_dir / "mutation_strategy_success.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    names = list(mut_scores.keys())
    means = [mut_scores[n] for n in names]
    colors = _PALETTE[: len(names)]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, means, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, means, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, min(1.05, max(means) * 1.4 + 0.1))

    _apply_style(
        ax,
        title=f"Mean Score by Mutation Strategy — {record.failure_mode.value}",
        xlabel="Mutator",
        ylabel="Mean Score",
    )
    fig.tight_layout()
    path = fig_dir / "mutation_strategy_success.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_category_comparison(records: list[RunRecord], fig_dir: Path) -> Path:
    """Bar chart of top attack score per failure category, across multiple run records."""
    labels = [r.failure_mode.value.replace("_", "\n") for r in records]
    top_scores = [
        r.top_attacks(1)[0].score.score if r.all_scored_prompts else 0.0
        for r in records
    ]
    colors = _PALETTE[: len(records)]

    fig, ax = plt.subplots(figsize=(max(6, len(records) * 2), 5))
    bars = ax.bar(labels, top_scores, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, top_scores, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, 1.15)

    _apply_style(
        ax,
        title="Top Attack Score by Failure Category",
        xlabel="Failure Mode",
        ylabel="Top Score",
    )
    fig.tight_layout()
    path = fig_dir / "category_success_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
