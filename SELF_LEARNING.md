# Self-Learning Multi-Agent LLM Training

A self-improving training loop with hierarchical agents, judges-of-judges, Hyperband
scheduling, Braintrust observability, and a live dashboard.

## Files

| File | Role |
|---|---|
| `agents.py` | Multi-agent hierarchy (Trainer → Evaluator → Judge → MetaJudge → Human) |
| `hyperband.py` | Adaptive scheduler + cheap-GP prior over the architecture space |
| `braintrust_bridge.py` | Sends traces to Braintrust + fetches them back for the dashboard |
| `self_learning_runner.py` | Main loop — glues everything together |
| `server_v3.py` | Flask backend serving the dashboard + APIs |
| `dashboard.html` | Live single-page dashboard |

## Architecture

```
                    ┌────────────────────┐
                    │ OrchestratorAgent  │  opus
                    └─────────┬──────────┘
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  ┌──────────┐         ┌────────────┐        ┌──────────────┐
  │ Trainer  │ haiku   │ Evaluator  │ haiku  │ HumanJudge   │  (queue)
  └──────────┘         └─────┬──────┘        └──────────────┘
                             │                       ▲
                             ▼                       │ escalates on bias
                       ┌──────────┐                  │
                       │  Judge   │ sonnet           │
                       └────┬─────┘                  │
                            ▼                        │
                       ┌─────────────┐               │
                       │ MetaJudge   │ opus ─────────┘
                       └─────────────┘
```

Every agent call → traced span in Braintrust + appended to local state file.

## Self-learning loop (one round)

1. **Trainer** proposes a config (LLM call, primed with prior sweep history).
2. Hyperband bracket runs N candidates with successive halving; bad ones killed early.
3. **Evaluator** scores the survivor (val_ppl, cloze_acc, sample quality).
4. **Judge** audits the Evaluator (sonnet + extended thinking) — accept/reject.
5. **MetaJudge** audits the Judge over time, looking for bias (lenient/strict/anchoring/sample-blindness).
6. If MetaJudge flags bias OR Judge confidence < 0.4 → escalate to **HumanJudge** queue.
7. Cheap-GP prior is updated with all candidates (not just winner) — next round proposes from a smarter prior.

## Run it

```bash
# 1. Install deps
pip install anthropic braintrust requests flask numpy torch tiktoken

# 2. Set keys (rotate the leaked one in settings.local.json first!)
export ANTHROPIC_API_KEY=sk-ant-...
export BRAINTRUST_API_KEY=sk-...     # optional — degrades gracefully without
export BRAINTRUST_PROJECT=llm-training-self-learning

# 3. Start the dashboard
python3 server_v3.py &
open http://localhost:8001

# 4. Start the self-learning loop
python3 self_learning_runner.py --rounds 4 --max-steps 100 --reset
```

## Why this is "self-learning"

Three independent learning signals stack:

1. **Hyperband** learns within a round which configs deserve more compute (kills bad ones early).
2. **CheapPrior (RBF-weighted)** learns across rounds which regions of architecture space are promising. It refits after every candidate, not just every round.
3. **Trainer agent** is given the full history each round and proposes the next config — it's literally Claude reading prior loss curves and reasoning about what to try.
4. **MetaJudge** learns about the Judge's bias over time and escalates to humans when it drifts.

Human decisions go back into the queue file, which can be read by future runs as
authoritative labels — closing the loop.

## Dashboard panels

- **Live system stats** — phase, round, best ppl, prior support size
- **Agent hierarchy** — visual tree, active agent pulses
- **Current winner** — config + sample (collapsible)
- **Loss curve** — all rounds overlaid, latest highlighted
- **Hyperband bracket** — candidate→halving→winner per round
- **Agent message stream** — live event tail from local state
- **Latest judge verdict** — accept/reject + meta-judge bias check
- **Braintrust traces** — pulled live via REST API (or local fallback)
- **Human-judge queue** — pending escalations
- **Round history** — full table

## Security

The `BRAINTRUST_API_KEY` was found in plaintext in `~/.claude/settings.local.json:59`.
Rotate it in the Braintrust dashboard and use the env var instead.
