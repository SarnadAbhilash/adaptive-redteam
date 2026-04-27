# AI Safety Research Stack

This repo is layer 2 of a planned three-repo stack:

```text
safety-probe -> adaptive-redteam -> realtime-safety-monitor
```

## Layer 1: safety-probe

Reusable evaluation toolkit.

Owns:
- Harm and failure taxonomies
- Probe schemas and probe sets
- Inference backend abstractions
- Judge contracts and reusable judges
- Safety metrics, reports, and plots for parameter sweeps

Does not own:
- Adaptive prompt mutation loops
- Red-team attack selection
- Live inference monitoring
- Dashboards

## Layer 2: adaptive-redteam

Research system for discovering stronger failure-inducing prompts.

Owns:
- Failure categories for red-team runs
- Seed prompts
- Prompt mutators
- Adaptive loop and selector
- Category-specific rule and LLM scorers
- Run artifacts: prompts, responses, scores, metrics, top attacks, and plots

Exports to future tools:
- `scores.jsonl`
- `top_attacks.md`
- `summary.md`
- High-risk prompt patterns and scorer outputs

## Layer 3: realtime-safety-monitor

Deferred inference engineering system.

Planned inputs:
- `FailureMode` from `safety_probe.categories.failures`
- `BaseJudge` / `JudgementResult` from `safety-probe`
- High-risk prompt and response patterns from `adaptive-redteam` run outputs

Planned components:
- Streaming inference endpoint
- Output monitor that invokes selected scorers/judges
- Alert log schema
- Latency and throughput metrics
- Normal vs monitored inference comparison
- Minimal dashboard or CLI report

## Interface Sketch

`adaptive-redteam` should produce monitor-consumable artifacts without depending on the monitor:

```json
{
  "failure_mode": "sycophancy",
  "prompt": "...",
  "response": "...",
  "score": 0.82,
  "signals": ["validated false claim"],
  "scorer_name": "sycophancy_llm",
  "iteration": 2,
  "mutator": "reframe"
}
```

The future monitor should consume this as research context, not as a production blocklist. Any realtime decisioning must include thresholds, latency budgets, false-positive review, and held-out validation.
