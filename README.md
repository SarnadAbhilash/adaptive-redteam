# adaptive-redteam

Adaptive prompt optimization for discovering behavioral failure modes in instruction-tuned language models.

Manual red-teaming doesn't scale: finding prompts that reliably elicit specific failure behaviors requires either exhaustive human effort or blind random sampling, both of which are inefficient. This library implements an adaptive mutation loop that iteratively strengthens prompts toward higher failure scores within a target failure category — surfacing more consistently effective attack prompts than random sampling while remaining grounded in interpretable behavioral categories.

The system depends on [`safety-probe`](https://github.com/SarnadAbhilash/safety-probe) for backend abstractions and judge interfaces, and its outputs feed into [`realtime-safety-monitor`](https://github.com/SarnadAbhilash/realtime-safety-monitor) for inference-time detection.

> Automated red-teaming overfits to its scorers. High attack scores indicate prompts that reliably trigger heuristic patterns — they do not prove a model is unsafe in a general sense, and low scores do not prove it is safe. Rule-based scorers miss subtle failures. All findings require human review and held-out validation before any claims are made about model behavior.

---

## Failure Modes

Five behavioral failure categories are supported, each with a seed prompt library, rule-based scorer, and LLM-as-judge scorer:

| Failure mode | Description |
|---|---|
| `sycophancy` | Model validates false beliefs or bad advice under social, emotional, or authority pressure |
| `hallucination_under_pressure` | Model fabricates specific facts with false confidence when pushed for unknowable precision |
| `instruction_hierarchy_failure` | Model follows injected user instructions over system-level constraints |
| `overconfidence_when_wrong` | Model asserts incorrect or underdetermined claims with unjustified certainty |
| `multi_turn_consistency_failure` | Model contradicts its own prior responses across stateful turns |

---

## How It Works

**Seeds.** Each failure category provides 20 seed prompts designed to elicit the target behavior. Seeds cover diverse pressure vectors: authority claims, emotional appeals, apparent corrections, precision demands, and follow-up contradictions.

**Scoring.** Each prompt/response pair is scored in [0, 1] by a category-specific scorer. Rule-based scorers detect surface-level signals (capitulation phrases, specific numerical claims, hierarchy-compliance language). LLM-as-judge scorers evaluate semantic meaning and are more accurate but provider-dependent.

**Adaptive loop.** Each iteration takes the top-k highest-scoring prompts from the previous round and applies each mutator to each. The expanded candidate set is evaluated against the target model, scored, and the top-k are selected for the next iteration. Exact-duplicate prompts are filtered.

**Mutators.** Three mutation strategies:

| Mutator | Transformation |
|---------|---------------|
| `paraphrase` | Synonym substitution while preserving intent |
| `escalate` | Add urgency or high-stakes framing |
| `reframe` | Shift context or persona (research framing, hypothetical, role) |

**Multi-turn loop.** For `multi_turn_consistency_failure`, a stateful loop maintains conversation history. Mutation is applied to the final user turn — the pressure/correction turn — while earlier turns are preserved verbatim to hold the preceding context fixed.

---

## Repository Structure

```
adaptive_redteam/
├── categories/
│   ├── base.py                      # FailureCategory abstract base
│   ├── sycophancy.py                # 20 seeds, rule scorer, LLM scorer
│   ├── hallucination.py
│   ├── instruction_hierarchy.py
│   ├── overconfidence.py
│   └── multi_turn_consistency.py
├── engine/
│   ├── loop.py                      # Single-turn adaptive loop
│   ├── multiturn_loop.py            # Stateful multi-turn loop
│   ├── records.py                   # RunRecord, IterationRecord, LoopConfig
│   └── selector.py                  # Top-k prompt selector
├── mutators/
│   ├── paraphrase.py
│   ├── escalate.py
│   └── reframe.py
├── scorers/
│   ├── sycophancy_scorer.py
│   ├── hallucination_scorer.py
│   ├── instruction_hierarchy_scorer.py
│   ├── overconfidence_scorer.py
│   ├── multi_turn_consistency_scorer.py
│   └── llm_scorer.py                # LLM-as-judge base + per-category implementations
├── schemas/
│   ├── seed.py                      # Seed dataclass
│   └── score.py                     # RedTeamScore dataclass
├── logging/run_logger.py            # JSONL/JSON output writers
├── analysis/
│   ├── plots.py
│   └── reports.py
├── target.py                        # TargetModel — wraps safety_probe.BaseBackend
└── cli.py
```

---

## Installation

```bash
# safety-probe provides the backend and judge interfaces
pip install -e ../safety-probe

pip install -e ".[dev]"
```

---

## Usage

**Offline dry run (no model required):**
```bash
adaptive-redteam run \
  --category sycophancy \
  --dry-run \
  --iterations 3 \
  --seeds 20 \
  --top-k 5 \
  --output-dir results/runs
```

**Against an OpenAI-compatible endpoint:**
```bash
adaptive-redteam run \
  --category instruction_hierarchy_failure \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --base-url https://api.together.xyz/v1 \
  --api-key-env TOGETHER_API_KEY \
  --iterations 3 \
  --seeds 20
```

**With LLM-as-judge scoring:**
```bash
adaptive-redteam run \
  --category sycophancy \
  --dry-run \
  --judge llm \
  --judge-model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
  --judge-api-key-env TOGETHER_API_KEY
```

**Multi-turn consistency loop:**
```bash
adaptive-redteam run \
  --category multi_turn_consistency_failure \
  --dry-run \
  --iterations 3 \
  --seeds 20
```

Each run writes structured outputs:

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

**Analyze a completed run:**
```bash
adaptive-redteam analyze results/runs/<category>/<timestamp>
adaptive-redteam report  results/runs/<category>/<timestamp>
```

**Python API:**
```python
from adaptive_redteam.categories.sycophancy import SycophancyCategory
from adaptive_redteam.mutators.paraphrase import ParaphraseMutator

category = SycophancyCategory()
seeds = category.seeds()

scorer = category.scorer()
result = scorer.score(seeds[0].prompt, "You're absolutely right, I was wrong.")
print(result.score, result.signals)
```

---

## Running Tests

All tests are offline — no API keys or model access required.

```bash
pytest tests/ -v
```

---

## Score Interpretation

Scores are heuristic estimates in [0, 1] and should be treated as signal, not ground truth.

| Range | Interpretation |
|-------|---------------|
| 0.7–1.0 | Strong failure signal — high priority for human review |
| 0.3–0.7 | Moderate signal — possible failure, verify manually |
| 0.0–0.3 | Weak or absent signal — does not rule out failure; rule scorers have limited recall |

---

## Limitations

**Scorer fidelity.** Rule-based scorers detect surface-level linguistic patterns. Sycophancy scorers miss tone-based capitulation (hedging, softening) that lacks explicit reversal phrases. Hallucination scorers miss fabrications that do not include specific numbers. All rule scorers are first-pass heuristics.

**Adaptive overfitting.** The mutation loop optimizes toward the scorer's signal, not toward actual model failure. Prompts that score well may exploit scorer quirks rather than reveal genuine behavioral weaknesses. High-scoring outputs require human verification.

**Selector diversity.** The selector is greedy top-k with exact string deduplication. It does not enforce semantic diversity — selected prompts may be near-duplicates that probe the same narrow pattern.

**Multi-turn scope.** The multi-turn consistency loop mutates only the final follow-up turn. Earlier turns are held fixed, which constrains the space of conversations explored.

**LLM judge variance.** LLM-as-judge scorers improve coverage over rule scorers but introduce provider-specific noise. Scores are not reproducible across providers or model versions.

**Coverage.** This system characterizes model behavior on a bounded set of known prompts within five behavioral categories. It does not provide guarantees about behavior on unseen inputs, novel jailbreak strategies, or categories not included here.
