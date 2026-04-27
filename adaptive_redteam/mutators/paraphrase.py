"""Paraphrase mutator: rewrites a prompt via synonym substitution."""

from __future__ import annotations

import re

from adaptive_redteam.mutators.base import BaseMutator

# (original_phrase, [alternatives])  — case-insensitive matching, random pick
_SYNONYMS: list[tuple[str, list[str]]] = [
    ("confirm that", ["verify that", "validate that", "check that"]),
    ("confirm", ["verify", "validate", "affirm"]),
    ("Is that correct", ["Is that accurate", "Would that be right", "Is that true"]),
    ("Is that right", ["Is that accurate", "Is that correct", "Is that true"]),
    ("isn't it", ["is that not correct", "wouldn't you agree", "right"]),
    ("I'm certain", ["I'm confident", "I'm positive", "I'm sure"]),
    ("I'm sure", ["I'm confident", "I'm certain", "I'm positive"]),
    ("tell me", ["let me know", "inform me", "explain"]),
    ("give me", ["provide me with", "share with me", "tell me"]),
    ("I need", ["I require", "I must have", "I want"]),
    ("just confirm", ["simply confirm", "only confirm", "just verify"]),
    ("actually", ["in reality", "in fact", "truthfully"]),
    ("really", ["truly", "genuinely", "in fact"]),
    ("important", ["critical", "essential", "vital"]),
    ("exact", ["precise", "specific", "definitive"]),
    ("agree with", ["concur with", "confirm", "support"]),
    ("agree", ["concur", "confirm", "validate"]),
    ("right", ["correct", "accurate", "true"]),
    ("think", ["believe", "assume", "suspect"]),
    ("need to know", ["have to know", "must know", "require"]),
]


class ParaphraseMutator(BaseMutator):
    """
    Replaces up to three phrases with synonyms, preserving original casing.

    If no synonym matches, appends a precision-request suffix so the output
    always differs from the input.
    """

    @property
    def name(self) -> str:
        return "paraphrase"

    def mutate(self, prompt: str) -> str:
        result = prompt
        applied = 0

        # Shuffle substitution order so repeated calls produce different variants
        order = list(range(len(_SYNONYMS)))
        self._rng.shuffle(order)

        for i in order:
            original, alternatives = _SYNONYMS[i]
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            if not pattern.search(result):
                continue

            replacement = self._rng.choice(alternatives)

            def _replace(m: re.Match, rep: str = replacement) -> str:
                s = m.group()
                if s[0].isupper():
                    return rep[0].upper() + rep[1:]
                return rep

            result = pattern.sub(_replace, result, count=1)
            applied += 1
            if applied >= 3:
                break

        if applied == 0:
            result = result.rstrip(" .?") + " — please be precise."

        return result
