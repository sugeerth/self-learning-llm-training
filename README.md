# Self-Learning LLM Training

> A self-learning multi-agent LLM training loop with hierarchical judge agents, Hyperband scheduling, and a live observability dashboard.

## Overview

A self-improving training loop where a hierarchy of agents — Trainer ->
Evaluator -> Judge -> MetaJudge — drives, evaluates, and critiques its own LLM
training runs. Hyperband adaptively allocates compute across the architecture
search space, while a live dashboard streams round history, judge verdicts,
and traces. The MetaJudge audits the Judge for bias and escalates close calls
to a human-judge queue.

## Features

- Hierarchical multi-agent loop: an Orchestrator coordinates Trainer, Evaluator, Judge, MetaJudge, and a Human-judge escalation queue.
- Judges-of-judges: the MetaJudge audits Judge verdicts for bias and escalates contested decisions to humans.
- Hyperband scheduling with successive halving and a cheap-GP prior over the architecture space.
- Live single-page observability dashboard served by Flask — round history, current phase, agent verdicts, and traces.
- Braintrust observability bridge with a local fallback when no API key is set.
- nanoGPT-style training, benchmarking, and inference utilities, including GRPO and speculative-decoding experiments.
- One-command pipeline: install, sweep variants, benchmark the winner, and serve the dashboard.

## Tech Stack

- Python
- PyTorch
- Flask (dashboard backend)
- Anthropic API (agent reasoning)
- Braintrust (optional observability)
- tiktoken, numpy

## Getting Started

```sh
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...               # agent reasoning
export BRAINTRUST_API_KEY=...              # optional — degrades gracefully without

# Run the self-learning loop
python3 self_learning_runner.py --max-steps 100 --rounds 4

# Serve the live dashboard (reads the snapshot written by the runner)
python3 server_v3.py            # http://localhost:8000
```

Or run the full pipeline end-to-end (install -> sweep -> benchmark -> serve):

```sh
./run.sh
```

## Throughput harness

`harness.py` is a drop-in execution engine for the Hyperband sweep that targets
candidate-evaluations/hour (see `10X_PLAN.md`, Pillar 2):

- **Parallel rungs** — a rung's candidates train concurrently across worker processes.
- **Checkpoint promotion** — survivors resume from checkpoints and train only the delta
  steps (from-scratch halving costs r + 2r + 4r per survivor path; promotion costs
  r + r + 2r — and the winner has genuinely accumulated training).
- **Adaptive thread split** — late rungs with fewer candidates than workers hand the
  idle cores to the remaining tasks.
- **SNIP proxy pre-filter** — optional zero-cost saliency ranking picks bracket entrants
  from an oversampled pool before any training steps are spent.
- **Self-optimization** — `harness.py tune` probes worker/thread splits with real
  training tasks, measures aggregate steps/sec, and persists the best profile to
  `harness_profile.json`; every later run starts at the machine's measured peak.

```sh
python3 harness.py tune     # self-optimize for this machine (once)
python3 harness.py bench    # baseline vs harness on the same bracket, prints speedup
python3 self_learning_runner.py --harness --rounds 4   # sweep through the harness
```

Measured on a 4-core CPU container (identical candidates, same bracket): **2.1–2.3x**
wall-clock, 184 -> 379 candidate-evals/hour (n=4) and 237 -> 539 (n=8). The tuner's own
probes put this box's parallel-efficiency ceiling at 1.63x — rung-0 parallelism equals
the candidate count, so the same harness scales toward ~10x with more cores or GPUs.
If tiktoken cannot fetch the GPT-2 vocab (offline/air-gapped), data prep degrades
gracefully to a byte-level tokenizer.

## Baseline arms (regret vs. random)

`arms.py` is the proof side (see `10X_PLAN.md`, Pillar 1): it runs **random search**,
**Hyperband**, **Hyperband + CheapPrior**, **evolve** (genetic search), and (with an
API key) **+ Trainer agent** at an identical training-step budget, scored by an
identical deterministic fixed-window eval, all executing through the throughput
harness. The headline is *steps to reach random search's final quality*, normalized
by the steps random itself needed — the single number that says whether the loop is
actually learning.

```sh
python3 arms.py run --quick      # 2 seeds x 96-step budget (~20 min on 4 CPU cores)
python3 arms.py run              # 3 seeds x 256-step budget
```

Outputs `arms_report.json` (trajectories + per-arm quality at 25/50/75/100% of budget)
and `arms_report.html` (self-contained SVG regret plot, no dependencies). Add
`--warm-prior prior_store.json` to let the prior arm compound knowledge across runs.

### Evolutionary arm — breeding, not just sampling

Every other arm *samples* the config space; `evolve.py` *breeds* it. A genome is a
hyperparameter config; fitness is the same deterministic val-ppl; the winners are
recombined by **uniform crossover + local mutation**, and each offspring is
**pre-screened by the CheapPrior surrogate** (the project's cheap config→ppl
predictor) so real training steps are spent only on children the surrogate likes.
That fuses the two existing pillars — the throughput harness and the prior — into the
genetic operators. The only difference from the `random` arm is where the next genome
comes from, so the same regret metric reads as a clean "does breeding beat sampling?".

```sh
python3 evolve.py run --budget 420 --full-steps 12 --pop 6 --elite 3   # standalone
python3 evolve_viz.py                                                   # -> evolve_report.html
```

`evolve.py` records every genome with its generation, parents and origin;
`evolve_viz.py` renders that as a **genealogy DAG** (height = fitness, edges =
parent→child) plus a **config-space view** of how the search concentrates over
generations — the shape of the search a regret curve summarises but hides. A real
10-generation, 33-genome run halved best val ppl versus the random generation 0.

## Flywheel experiment (does self-generated data help?)

`flywheel.py` turns the synthetic-data claim into a paired A/B: a generator model is
trained through the harness, generates continuations from real-prefix seeds, survivors of
a filter are mixed into a real-data subset at each ratio, and identically-seeded models
are trained on real-only vs mixed corpora and compared on the real validation set.
Filtering is offline-heuristic by default (degenerate-repetition, low-diversity, and
near-duplicate gates); with `ANTHROPIC_API_KEY` set, the 3-judge Claude ensemble in
`synthetic_flywheel.py` takes over.

```sh
python3 flywheel.py run --quick   # ~10 min on 4 CPU cores -> flywheel_report.json
```

Each ratio gets a verdict: **gain / neutral / collapse** (±2% relative val loss).

## Architecture

```
                    ┌────────────────────┐
                    │ OrchestratorAgent  │
                    └─────────┬──────────┘
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  ┌──────────┐         ┌────────────┐        ┌──────────────┐
  │ Trainer  │         │ Evaluator  │        │ HumanJudge   │  (queue)
  └──────────┘         └─────┬──────┘        └──────────────┘
                             │                       ▲
                             ▼                       │ escalates on bias
                       ┌──────────┐                  │
                       │  Judge   │                  │
                       └────┬─────┘                  │
                            ▼                        │
                       ┌─────────────┐               │
                       │ MetaJudge   │ ──────────────┘
                       └─────────────┘
```

Every agent call becomes a traced span (Braintrust + a local event stream).

## Key Files

| File | Role |
|---|---|
| `agents.py` | Multi-agent hierarchy (Trainer -> Evaluator -> Judge -> MetaJudge -> Human) |
| `hyperband.py` | Adaptive scheduler + cheap-GP prior over the architecture space |
| `braintrust_bridge.py` | Sends traces to Braintrust + fetches them back for the dashboard |
| `self_learning_runner.py` | Main loop — glues everything together |
| `server_v3.py` | Flask backend serving the dashboard + APIs |
| `dashboard.html` | Live single-page dashboard |
| `model.py` / `train.py` | nanoGPT-style model + training loop |
| `benchmark.py` / `inference.py` | Evaluation and inference utilities |

## Security

API keys are read from environment variables only — never hardcode them.
Rotate any key that has been exposed and set it via the env var instead.

## License

MIT — see [LICENSE](LICENSE).
