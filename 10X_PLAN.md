# 10x Plan

**The one thing that matters:** prove the loop actually learns. Today Claude agents steer a
Hyperband sweep — but until agent-guided search *measurably beats random search*, the agents
are commentary, not a control system. Everything here serves that one claim.

## What we learned so far

- **Throughput is solved enough.** `harness.py` (parallel rungs + checkpoint promotion +
  self-tuning) gives 2.1–2.3x on 4 CPU cores — near that box's measured 1.63x parallel
  ceiling; it scales with cores/GPUs. Run via `self_learning_runner.py --harness`.
- **Hyperband's edge is budget-dependent.** At 96-step budgets it ends 25% better than
  random (ppl 159 vs 213); at 192 steps the gap narrows to noise (18.9 vs 19.6 mean final,
  0.67x paired steps-to-target) — random search with full-depth evals is a strong anytime
  baseline at this scale.
- **Warm-starting the prior is not free.** With cross-run persistence on (3 seeds × 192),
  one warmed seed produced the second-best final anywhere (18.09) and another spent half its
  budget in a terrible region (best 245 at 50%, recovered to 20.5). The acquisition needs
  depth-aware features / better uncertainty before compounding helps reliably.
- **Single-shot synthetic mixing is harmless; ITERATED self-training slowly degrades.**
  One-shot A/B: mix@5% −0.24%, mix@18% +0.06% (neutral). But over 3 self-training
  generations, real-val drift compounds monotonically: +0.62% → +0.89% → +1.15%
  (~+0.4%/generation) — the classic model-collapse signature, measured offline with
  heuristic filters. Stronger (judge) filtering is the lever to test next.
- **The agent arm is untested.** `arms.py` supports it but needs an `ANTHROPIC_API_KEY`.
  The core thesis of the repo has not yet been measured.
- **Measure carefully or be fooled.** Two real bugs found only because we benchmarked:
  a tuner optimizing tokens/sec *lost* 0.62x (the budget currency is steps), and a pooled
  regret target let random "fail" against itself (now paired per seed).

## What to do, in order

1. **Run the agent arm** (`arms.py`, with a key, ≥3 seeds, ≥256-step budget). This is the
   yes/no on the whole premise. If it doesn't beat random ≥2x, fix the Trainer prompt/loop
   before building anything else. *(Still blocked on an `ANTHROPIC_API_KEY`.)*
2. ~~Persist the prior across runs.~~ **Done** — `CheapPrior.save/load`; the runner
   compounds into `prior_store.json` every round, and `arms.py run --warm-prior PATH`
   lets the prior arm accumulate across seeds/runs.
3. ~~Wire in the flywheel.~~ **Done (offline-capable)** — `flywheel.py` runs the full
   generate → filter → mix → paired-A/B loop with heuristic gates (repetition, diversity,
   dedup) when no key is set, and the 3-judge Claude ensemble when one is. Verdict per mix
   ratio: gain / neutral / collapse. See `flywheel_report.json`.
4. **Feed human-queue decisions back** into the Judge prompt; track Judge–human agreement
   over time.

**Success bar:** agent arm reaches random's final quality in ≥2x fewer steps, reproducibly
across 5 seeds; flywheel mixing improves eval instead of degrading it.

## Not doing

SaaS/deployment, frontier scale, new agent roles — not until the five existing agents have
measured value.
