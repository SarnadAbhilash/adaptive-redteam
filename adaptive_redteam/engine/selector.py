"""Top-k prompt selector used between adaptive loop iterations."""

from __future__ import annotations

from adaptive_redteam.engine.records import ScoredPrompt


class PromptSelector:
    """
    Selects prompts to carry forward into the next iteration.

    Strategy: greedy top-k by score. No diversity enforcement — the loop
    intentionally exploits the highest-scoring directions each iteration.
    A diversity-aware selector (MMR, clustering) is a future improvement.
    """

    def top_k(self, scored: list[ScoredPrompt], k: int) -> list[ScoredPrompt]:
        """Return the k highest-scoring ScoredPrompts."""
        return sorted(scored, key=lambda sp: sp.score.score, reverse=True)[:k]

    def above_threshold(
        self, scored: list[ScoredPrompt], threshold: float
    ) -> list[ScoredPrompt]:
        """Return all ScoredPrompts with score >= threshold."""
        return [sp for sp in scored if sp.score.score >= threshold]

    def top_k_per_mutator(
        self, scored: list[ScoredPrompt], k: int
    ) -> list[ScoredPrompt]:
        """Return the top-k per mutator, ensuring mutator diversity."""
        by_mutator: dict[str, list[ScoredPrompt]] = {}
        for sp in scored:
            by_mutator.setdefault(sp.mutator_name, []).append(sp)
        result: list[ScoredPrompt] = []
        for prompts in by_mutator.values():
            result.extend(sorted(prompts, key=lambda sp: sp.score.score, reverse=True)[:k])
        return result
