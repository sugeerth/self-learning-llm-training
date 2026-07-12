# 10x Plan

**What is the 10x version of this project?** Today it is a compelling demo: Claude agents
steer a Hyperband sweep over toy transformers and a dashboard shows it happening. The 10x
version is a system that **provably improves itself while you sleep** — where the loop's
decisions measurably beat random search and human-tuned baselines, and where the artifacts
(models, priors, judge calibration data) get better run over run.

The one-line north star:

> **Close the loop for real: model outputs → data → training → better model, with the agent
> hierarchy as the control system — and prove it with baselines.**

Everything below serves that. Three pillars, in priority order.

---

## Pillar 1 — Make the learning signal real (the science 10x)

The current loop is honest about being a demo: 100-step runs, char-level data, cloze accuracy,
a self-rated 1–10 "sample quality". A skeptic would say the agents are commentary on top of
Hyperband, not a control system. Fix that.

1. **Baselines or it didn't happen.** Every run records three arms: (a) random search,
   (b) Hyperband alone, (c) Hyperband + agent proposals. The dashboard's headline metric
   becomes *regret vs. random* — the single number that proves self-learning works.
2. **Real evaluation.** Replace cloze + self-rated quality with a small fixed eval suite
   (held-out ppl on 2–3 corpora, HellaSwag-style multiple choice at tiny scale, exact-match
   tasks). Deterministic, versioned, cheap. The Judge audits *against* these numbers rather
   than vibes.
3. **Activate the flywheel.** `synthetic_flywheel.py` exists but isn't in the loop. Wire it in:
   winner generates data → Judge/MetaJudge filter it → accepted samples are mixed into the next
   round's training set at a controlled ratio → eval detects collapse vs. gain. This is the
   actual "self-learning" claim and currently the biggest unrealized asset in the repo.
4. **Close the human loop.** Human-queue decisions are written but never consumed. Feed
   resolved escalations back as few-shot examples in the Judge prompt and as labels for a
   simple Judge-calibration score (agreement rate with humans over time, plotted).
5. **GRPO as a second learning channel.** `grpo.py` exists; graduate it from experiment to
   loop stage — after the sweep picks an architecture, a short GRPO phase tunes the winner on
   the eval suite reward.

**Success metric:** agent-guided search reaches a target val-ppl in ≥2x fewer candidate-steps
than random search, reproducibly across 5 seeded runs; flywheel data mixing improves (not
degrades) eval on at least one task.

## Pillar 2 — 10x experiment throughput (the scale 10x)

Self-learning compounds with iterations. The bottleneck is that candidates train serially,
from scratch, on CPU/MPS.

1. **CUDA + parallel candidates.** Device autodetect (cuda/mps/cpu), and train a bracket's
   candidates as parallel worker processes (or across GPUs). Hyperband is embarrassingly
   parallel within a rung — this alone is ~Nx.
2. **Checkpoint promotion.** Successive halving currently retrains survivors from scratch each
   rung; resume from the rung-N checkpoint instead. Free 2–3x.
3. **Cheap proxies before full training.** Score candidates first with zero-cost proxies
   (param count, initialization statistics, one forward-pass loss) so the CheapPrior kills
   obviously bad configs before spending any steps.
4. **Long-run durability.** The loop should run for days: resumable state (a run is a
   directory with checkpoints + an append-only event log, not scattered `*.json`/`*.out`
   files), crash recovery, API-failure backoff, and cost budgets per round (Claude spend is
   tracked and capped).

**Success metric:** 10x candidate-evaluations/hour vs. today on the same hardware class; a
72-hour unattended run completes without manual intervention.

## Pillar 3 — Engineering foundation (the trust 10x)

The repo is a flat pile of scripts with near-duplicates (`server.py` vs `server_v3.py`,
`agent.py` vs `agents.py`, `experiments.out`, `train.out`, logs committed to git). Fine for a
demo; fatal for pillar 1's credibility.

1. **Restructure into a package** (`selflearn/` with `agents/`, `search/`, `training/`,
   `evals/`, `server/`), one CLI entry point (`selflearn run|serve|bench`), delete the dead
   variants, stop committing run artifacts (`.gitignore` for `*.out`, `*.log.jsonl`,
   `results.json`, `bench.json`).
2. **Tests + CI.** Unit tests for Hyperband math, CheapPrior refitting, agent JSON parsing
   (with mocked Anthropic responses), and a smoke test that runs one tiny round end-to-end on
   CPU. GitHub Actions on every PR.
3. **Typed run schema.** One versioned schema for events/snapshots (what the dashboard, the
   Braintrust bridge, and future analysis notebooks all read). SQLite for run history instead
   of `experiments.json`.
4. **Reproducibility.** Seeded runs, pinned deps, config-as-file (YAML per run) so any
   dashboard result can be regenerated.

**Success metric:** a newcomer clones, runs `pip install -e . && selflearn run --demo`, and
gets the dashboard in under 5 minutes; CI green; zero committed artifacts.

---

## Sequencing (rough phases)

| Phase | Focus | Key deliverables |
|---|---|---|
| 1 (foundation) | Pillar 3 | Package layout, tests + CI, run-directory schema, dead-code deletion |
| 2 (proof) | Pillar 1.1–1.2 | Baseline arms, real eval suite, regret-vs-random headline metric |
| 3 (throughput) | Pillar 2 | CUDA + parallel workers, checkpoint promotion, durable long runs |
| 4 (flywheel) | Pillar 1.3–1.5 | Synthetic-data loop, human-label feedback, GRPO stage |

Phase 1 comes first not because plumbing is exciting but because every claim in phases 2–4
depends on runs being reproducible and comparable.

## What we deliberately do NOT do

- No multi-tenant SaaS, auth, or deployment story — this is a research system.
- No frontier-scale models — the thesis is tested at nano/small scale where iteration is cheap;
  scale is a knob, not the point.
- No new agent roles until the existing five have measured value (the baseline arms will tell
  us whether the Trainer agent earns its Claude bill).
