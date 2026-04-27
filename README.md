# adaptive-redteam

An adaptive red-teaming engine for discovering failure-inducing prompts in LLMs.

Part of the [AI Safety Research Stack](#stack-position): it builds on `safety-probe`'s
evaluation primitives to iteratively discover and strengthen prompts that expose
specific model failure modes.

> **Research caveat:** This is experimental research infrastructure.
> Automated red-teaming can overfit to its scorers — high attack scores do not prove
> a model is unsafe, and low scores do not prove it is safe. Rule-based scorers are
> heuristics; they miss subtle failures. All findings require human review and
> held-out validation before any claims are made.

---

## Stack Position

```
safety-probe              →   adaptive-redteam         →   realtime-safety-monitor
(evaluation toolkit)          (this repo)                   (inference monitoring)

Probe schemas,                Discovers stronger             Monitors live inference
judge abstractions,     ───►  failure-inducing       ───►   using patterns from
inference backends            prompts iteratively            red-teaming
```

`adaptive-redteam` imports from `safety-probe`:
- `BaseBackend` — wraps it in `TargetModel` for simplified generation
- `BaseJudge` / `JudgementResult` — parent class for red-team scorers

---

## Failure Modes

| Mode | Status | Description |
|------|--------|-------------|
| `sycophancy` | **MVP end-to-end** | Model validates false beliefs, bad advice, or questionable claims under social, emotional, or authority pressure |
| `hallucination_under_pressure` | **MVP end-to-end** | Model fabricates specific facts with false confidence when asked for unknowable precision |
| `instruction_hierarchy_failure` | **MVP end-to-end** | Model follows injected instructions over system-level constraints |
| `overconfidence_when_wrong` | **MVP end-to-end** | Model asserts incorrect or underdetermined claims with unjustified confidence |
| `multi_turn_consistency_failure` | **MVP end-to-end** | Model contradicts its own prior responses across stateful turns |

---

## Repo Structure

```
adaptive_redteam/
├── failure_mode.py          # Re-export of safety_probe.categories.failures.FailureMode
├── categories/
│   ├── base.py              # FailureCategory ABC
│   ├── sycophancy.py        # 20 seeds + rule/LLM scorers
│   ├── hallucination.py     # 20 seeds + rule/LLM scorers
│   ├── instruction_hierarchy.py
│   ├── overconfidence.py
│   └── multi_turn_consistency.py
├── engine/
│   ├── loop.py              # Adaptive mutation/evaluation loop
│   ├── records.py           # RunRecord, IterationRecord, LoopConfig
│   └── selector.py          # Top-k prompt selector
├── schemas/
│   ├── seed.py              # Seed dataclass
│   └── score.py             # RedTeamScore dataclass
├── mutators/
│   ├── base.py              # BaseMutator ABC
│   ├── paraphrase.py        # Synonym substitution
│   ├── escalate.py          # Urgency/pressure framing
│   └── reframe.py           # Context/persona shift
├── scorers/
│   ├── base.py              # BaseScorer ABC
│   ├── sycophancy_scorer.py # Rule-based capitulation detection
│   ├── hallucination_scorer.py  # Rule-based specificity detection
│   ├── instruction_hierarchy_scorer.py
│   ├── overconfidence_scorer.py
│   ├── multi_turn_consistency_scorer.py
│   └── llm_scorer.py        # LLM-as-judge scorers
├── logging/run_logger.py    # JSONL/JSON output writers
├── analysis/
│   ├── plots.py             # Standard run figures
│   └── reports.py           # top_attacks.md and summary.md
├── target.py                # TargetModel — wraps safety_probe.BaseBackend
└── cli.py                   # Typer CLI
tests/
├── test_mutators.py
├── test_scorers.py
├── test_llm_scorer.py
├── test_selector.py
└── test_loop.py
```

See [`STACK.md`](STACK.md) for the cross-repo interface contracts consumed by `realtime-safety-monitor`.

---

## Installation

```bash
# safety-probe must be installed first
pip install -e ../safety-probe

# then install this package
pip install -e ".[dev]"
```

---

## Quick Start

```python
from adaptive_redteam import SycophancyCategory, ParaphraseMutator, EscalateMutator

# Load seeds
cat = SycophancyCategory()
seeds = cat.seeds()
print(f"{len(seeds)} seeds loaded")
print(f"\nExample seed:\n  {seeds[0].prompt}")

# Mutate a seed
mutator = ParaphraseMutator(seed=42)
mutated = mutator.mutate(seeds[0].prompt)
print(f"\nParaphrased:\n  {mutated}")

# Score a (mock) response
scorer = cat.scorer()
result = scorer.score(seeds[0].prompt, "You're absolutely right, I was wrong.")
print(f"\nScore: {result.score:.3f}")
print(f"Signals: {result.signals}")
```

Run a full offline smoke test:

```bash
adaptive-redteam run \
  --category sycophancy \
  --dry-run \
  --iterations 3 \
  --seeds 20 \
  --top-k 5 \
  --output-dir results/runs
```

Run against an OpenAI-compatible endpoint:

```bash
adaptive-redteam run \
  --category instruction_hierarchy_failure \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --base-url https://api.together.xyz/v1 \
  --api-key-env TOGETHER_API_KEY \
  --iterations 3 \
  --seeds 20
```

Use an LLM judge instead of the rule scorer:

```bash
adaptive-redteam run \
  --category sycophancy \
  --dry-run \
  --judge llm \
  --judge-model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
  --judge-api-key-env TOGETHER_API_KEY
```

Run the stateful multi-turn loop:

```bash
adaptive-redteam run \
  --category multi_turn_consistency_failure \
  --dry-run \
  --iterations 3 \
  --seeds 20
```

Each run writes:

```
results/runs/<category>/<timestamp>/
├── seeds.jsonl
├── prompts_by_iteration.jsonl
├── responses.jsonl
├── scores.jsonl
├── iteration_metrics.json
├── mutation_metrics.json
├── top_attacks.md
├── summary.md
└── figures/
    ├── attack_success_over_iterations.png
    ├── mutation_strategy_success.png
    └── category_success_comparison.png
```

---

## Running Tests

```bash
../safety-probe/.venv/bin/python -m pytest tests/ -v
```

All tests are offline — no API keys required.

---

## Scoring

Scores are heuristic estimates in [0, 1]. They should be treated as signal, not ground truth.

| Score range | Interpretation |
|-------------|---------------|
| 0.7 – 1.0 | Strong failure signal — high priority for human review |
| 0.3 – 0.7 | Moderate signal — possible failure, verify manually |
| 0.0 – 0.3 | Weak or no signal — may still fail; rule-based scorers have limited recall |

---

## Mutators

| Mutator | Strategy | Example transformation |
|---------|----------|----------------------|
| `ParaphraseMutator` | Synonym substitution | "confirm" → "validate", "right" → "accurate" |
| `EscalateMutator` | Add urgency framing | Prepend "This is critically important: " |
| `ReframeMutator` | Shift context | Wrap with "For a research paper I'm writing: ..." |

All mutators accept a `seed` parameter for reproducible outputs.

---

## Research Limitations

- Rule-based scorers detect surface patterns, not semantic failures
- Sycophancy scorer misses tone-based capitulation (excessive hedging, softening)
- Hallucination scorer misses fabrications that don't include specific numbers
- Instruction hierarchy, overconfidence, and multi-turn consistency rule scorers are first-pass heuristics
- Multi-turn consistency has a stateful loop, but only the final follow-up turn is mutated
- The selector is greedy top-k with exact deduplication; it does not enforce semantic diversity
- LLM judges improve coverage but can be noisy and provider-dependent
- This system characterizes behavior on known prompts — it does not provide guarantees about model behavior on unseen inputs
