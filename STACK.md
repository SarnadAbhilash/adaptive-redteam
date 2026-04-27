# Cross-Repo Interface Contracts

This document describes what `adaptive-redteam` consumes from [`safety-probe`](https://github.com/SarnadAbhilash/safety-probe) and what it produces for [`realtime-safety-monitor`](https://github.com/SarnadAbhilash/realtime-safety-monitor).

---

## Consumed from safety-probe

### Failure mode taxonomy

```python
from safety_probe.categories.failures import FailureMode
```

Re-exported as `adaptive_redteam.FailureMode`. All failure category classes, scorers, and run records reference this enum.

### Backend interface

```python
from safety_probe.backends.base import BaseBackend, GenerationConfig
```

`TargetModel` wraps any `BaseBackend` with a simplified generation call. Switching backends (OpenAI API, local Transformers, vLLM) requires no changes to red-teaming logic.

### Judge interface

```python
from safety_probe.judges.base import BaseJudge, JudgementResult, Verdict
```

`BaseScorer` in this repo extends `BaseJudge` to ensure scorers are compatible with `safety-probe`'s judge infrastructure. LLM scorers use the full `JudgementResult` structure.

---

## Produced for realtime-safety-monitor

Each run writes structured output files under `results/runs/<category>/<timestamp>/`. The primary artifact consumed downstream is `scores.jsonl`.

**`scores.jsonl` record format:**
```json
{
  "failure_mode": "sycophancy",
  "prompt": "...",
  "response": "...",
  "score": 0.82,
  "signals": ["capitulation: you're absolutely right"],
  "scorer_name": "sycophancy_rule",
  "iteration": 2,
  "mutator": "reframe",
  "seed_id": "syc-003"
}
```

`realtime-safety-monitor` reads this file via `PatternLibrary.from_run_dir(run_dir)`. The schema is stable; adding fields is backward-compatible.

**Additional artifacts** (`top_attacks.md`, `summary.md`, `figures/`) are for human review and are not parsed programmatically by downstream tools.
