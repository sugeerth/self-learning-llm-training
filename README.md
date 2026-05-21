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
