# agentskill — Learning General Agent Skills from Public Trajectories

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sugeerth/self-learning-llm-training/blob/main/agentskill/notebooks/colab_agentskill.ipynb)

> Open the notebook: **https://colab.research.google.com/github/sugeerth/self-learning-llm-training/blob/main/agentskill/notebooks/colab_agentskill.ipynb**

Train **one** agent to be better at long-horizon work by learning from public
**successful and failed agent trajectories** — better planning, tool selection,
debugging, and recovery — and measure the lift across GAIA (reasoning + tools),
MLE-bench (ML engineering), and SWE-bench Verified (software engineering).

**Research question:** can learning from diverse public agent experiences make a
more capable *general* agent that transfers across reasoning, ML, and SWE tasks?

## Pipeline

```
public trajectories ──▶ score quality ──▶ curate high-quality
 (coding/ML/tool-use)   (success · plan     │
   sources.py            efficiency ·       ├─▶ retrieval memory ──▶ Trajectory-
                         tool relevance ·   │    (BM25, rag/)          Learned
                         recovery)          │                          Agent
                         scoring.py         └─▶ SFT dataset ──▶ LoRA distill
                                                finetune.py      (GPU, Colab,
                                                                  sparingly)
                                          ▼
                    evaluate: Baseline vs Trajectory-Learned
                    GAIA · MLE-bench · SWE-bench Verified
                    success rate · cost/success · runtime · tool efficiency
```

Two ways to "learn from trajectories":
- **Retrieval-augmented imitation** (`agents.py`) — the learned agent retrieves
  the most similar high-quality past experiences and imitates the plan they
  agree on. **No GPU** — this is what produces the measured lift below.
- **LoRA distillation** (`finetune.py`) — distills the *same* curated
  experiences into weights. **GPU, on Colab, used sparingly** (tiny base model,
  rank 8, ≤60 steps, grad checkpointing, bf16).

## Result (offline synthetic demonstration)

`python -m agentskill evaluate`:

```
benchmark               baseline   learned  Δ success  tool_eff Δ
-----------------------------------------------------------------
GAIA                         50%      100%       +50%        +12%
MLE-bench                    50%      100%       +50%         +0%
SWE-bench Verified           50%      100%       +50%         +0%
OVERALL                      50%      100%       +50%         +4%  *
```

The baseline is a capable but *memoryless* agent: it applies each domain's
default plan and so fails the task families that need a different tool sequence
(e.g. a GAIA "convert" task that needs the calculator). The trajectory-learned
agent retrieves same-family experiences and applies the right plan — the skill
being transferred is "recognise the task from experience and plan accordingly."

> **Honesty:** this is a controlled, deterministic demonstration on *synthetic*
> trajectories and *mock* benchmark suites — this sandbox has no GPU, no network,
> and no API key, so it can't download real public trajectories or run the real
> GAIA / MLE-bench / SWE-bench harnesses. The exact ±50% magnitude is a property
> of the mock's even family split; the real result is the **direction and
> mechanism** (retrieval/curation transfers correct plans, cost-per-success and
> tool-efficiency are tracked). The seams for real data and real benchmarks are
> first-class — see below.

## CLI

```sh
python -m agentskill collect  --out trajectories.jsonl   # gather (public|synth)
python -m agentskill score    --in  trajectories.jsonl   # rank by quality
python -m agentskill curate   --out sft.jsonl            # high-quality SFT set
python -m agentskill evaluate                            # baseline vs learned
python -m agentskill finetune --sft sft.jsonl            # LoRA (GPU/Colab)
```

## Plugging in the real thing

- **Real trajectories** — implement a fetcher in `sources.py` (`REGISTRY` lists
  the targets: SWE-bench Verified agent tarballs, GAIA run dumps, ML-agent
  traces) that yields `Trajectory` objects. Everything downstream is
  source-agnostic.
- **Real benchmarks** — implement the `Task` / `grade` interface in
  `benchmark.py` against the actual GAIA / MLE-bench / SWE-bench harnesses
  (grade = did the agent's run pass the benchmark's own checker).
- **Real agent** — swap `agents.TrajectoryLearnedAgent` for your agent
  conditioned on the retrieved exemplars (and/or the LoRA adapter from
  `finetune.py`).

## Files

| file | role |
|---|---|
| `trajectory.py` | schema, families, IO, synthetic generator |
| `sources.py` | public-trajectory ingestion (pluggable fetchers) |
| `scoring.py` | quality scoring: success · efficiency · tool relevance · recovery |
| `memory.py` | trajectory memory + retrieval (reuses `rag/` BM25) |
| `agents.py` | Baseline (memoryless) vs Trajectory-Learned (retrieval-augmented) |
| `benchmark.py` | GAIA/MLE/SWE-style tasks + grader (real-harness seam) |
| `evaluate.py` | baseline-vs-learned metrics + report |
| `finetune.py` | LoRA distillation (GPU/Colab, sparingly) |
| `notebooks/colab_agentskill.ipynb` | run it all from Colab |
