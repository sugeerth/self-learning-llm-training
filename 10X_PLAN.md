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
- **Depth-aware features fixed the prior.** v1 (full-depth evals only) starved and warm-
  starting it was pathological: one seed burned half its budget at ppl 245. v2 attaches
  training depth as a feature so every rung eval contributes; re-run on the same
  seeds/budget: 0/3 reached → **3/3 reached at 1.75x**, and the best half-budget quality
  of any arm (20.8 vs hyperband's 22.9, random's 28.8). Measure → fix → re-measure works.
- **Iterated self-training collapse is driven by the GROWING synthetic share.**
  Accumulate mode drifted +0.62% → +0.89% → +1.15% (compounding). Replace mode (constant
  10% share) + a self-perplexity gate: +0.69% → +0.67% → **+0.49%** — flat, a small
  constant tax instead of compounding decay. Constant-share self-training is stable;
  single-shot mixing remains neutral (mix@5% −0.24%, mix@18% +0.06%).
- **The agent arm is untested.** `arms.py` supports it but needs an `ANTHROPIC_API_KEY`.
  The core thesis of the repo has not yet been measured.
- **Measure carefully or be fooled.** Two real bugs found only because we benchmarked:
  a tuner optimizing tokens/sec *lost* 0.62x (the budget currency is steps), and a pooled
  regret target let random "fail" against itself (now paired per seed).

## What to do, in order

1. **Run the agent arm** (`arms.py`, with a key, ≥3 seeds, ≥256-step budget). This is the
   yes/no on the whole premise. If it doesn't beat random ≥2x, fix the Trainer prompt/loop
   before building anything else. *(Still blocked on an `ANTHROPIC_API_KEY`.)*
2. ~~Persist the prior across runs.~~ **Done, then fixed** — depth-aware `CheapPrior` v2
   (every eval contributes, weighted by depth); persistence stores raw triples; the runner
   compounds into `prior_store.json`; `arms.py run --warm-prior PATH`. Re-measured: 1.75x,
   best anytime curve.
3. ~~Wire in the flywheel.~~ **Done, collapse lever found** — `flywheel.py run` (single-shot
   A/B) and `flywheel.py generations --mode accumulate|replace [--ppl-gate]` (iterated).
   Replace mode + ppl gate flattens the collapse curve (+1.15% → +0.49% at G3).
4. **Feed human-queue decisions back** into the Judge prompt; track Judge–human agreement
   over time.

**Success bar:** agent arm reaches random's final quality in ≥2x fewer steps, reproducibly
across 5 seeds; flywheel mixing improves eval instead of degrading it.

## Not doing

SaaS/deployment, frontier scale, new agent roles — not until the five existing agents have
measured value.

## 100x: measured (harness levers A/B/C)

Same bracket (n=4, halvings=2, initial_steps=4), same seeds, 4-core CPU —
`python3 harness.py bench100`, results in `bench100_harness.json`:

| Measurement | Result |
|---|---|
| Baseline sweep (serial, from-scratch, sample/eval) | 94.2s |
| Cold harness sweep (parallel + promote + kill) | 47.1s (**2.0x**, better ppl: 300.9 vs 324.1) |
| Warm identical re-sweep (eval cache, all hits) | 0.001s (**67,210x**) |
| Campaign of 10 sweeps | **20.0x** |
| Campaign of 50 sweeps | **99.8x ≈ the 100x** |
| Kill switch on a diverging population (lr=0.05) | 2/4 killed, **37% of budget refunded** |

The levers:
- **A. Eval cache** (`EvalCache`, content-addressed evals + checkpoints under
  `runs/cache/`): identical (config, lr, batch, steps, data, val) work never
  reruns — across rungs, arms, repeated sweeps, and crashes. Deployed in
  `self_learning_runner --harness` and `harness.py run` (off in `arms.py` so
  paired-budget arm comparisons stay honest).
- **B. Divergence early-kill** (`should_kill`, rolling-loss factor 2.5 after an
  8-step grace): doomed candidates stop billing the bracket mid-rung and rank
  last. Deployed in the runner and arms (refunds are budget-accounted).
- **C. Config dedupe**: colliding random draws train once.

Honesty note: a single cold sweep is bounded by cores (2.0x here); the 100x is
*effective* throughput on repeated/overlapping campaigns — exactly the arms /
regression-sweep / re-run-after-crash workloads this repo actually runs.

### Round 2: cold-sweep 10x levers (D: vocab clamp, E: bf16 autocast)

The cold sweep was core-bound at 2.0x. Two per-step levers fixed that
(`bench100` re-run, same bracket/seeds/machine):

- **D. Effective-vocab clamp**: the byte-level corpus contains ~123 distinct
  token ids, but configs allocated a 50,304-row embedding + output head — the
  dead-logit matmul+softmax dominated CPU step time. `effective_vocab()` scans
  the data and clamps candidates to the smallest covering 64-multiple (128
  here). Measured **4.8x per step** at d_model 512. Lossless for the task;
  ppl scale changes (smaller softmax denominator), so cross-engine ppl isn't
  comparable — rankings are, and all arms share one scale.
- **E. bf16 autocast (CPU)**: measured **1.66x per step**, and the numerics
  guard shows **0.0% val-ppl delta** vs fp32 on an identical seeded run
  (validation always runs fp32).

| Measurement | Round 1 | Round 2 |
|---|---|---|
| Cold harness sweep (vs 93.6-94.2s baseline) | 47.1s (2.0x) | **17.3s (5.4x)** |
| Campaign of 10 sweeps | 20.0x | **54.0x** |
| Campaign of 50 sweeps | 99.8x | **269x** |
| Sweeps to reach 100x | 50 | **~19** |
| bf16 numerics guard | — | 0.0% delta |

Both levers are default-on in `parallel_halving` (opt out: `auto_vocab=False`,
`amp=False`) and keyed into the eval cache, so fp32/amp and clamped/unclamped
results never collide. Historical arms JSONs predate the ppl-scale change —
re-baseline before comparing.
