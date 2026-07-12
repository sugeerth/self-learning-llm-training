# model-onramp

> Models are plugins. Infrastructure is permanent.

**model-onramp** is a model-onboarding layer for LLM infrastructure. When a new
model ships — a new Claude, an open-weights release, a fine-tune of your own —
you add **one ~20-line adapter file**. Everything built on top (training loops,
evals, judges, benchmarks, dashboards) picks it up automatically, with zero
changes to the infrastructure itself.

## The idea

Most LLM stacks hard-code their models: model names in config files, provider
SDKs called directly from training code, capability assumptions (context
length, tool use, JSON reliability) baked into prompts and schedulers. Every
new model release means a sweep through the whole codebase.

model-onramp inverts that:

1. **One-file adapters — or zero.** A new model enters the system as a single
   file subclassing `AdapterBase` — implement `_complete()`, declare pricing,
   done. Timing, token accounting, and cost come free. Adapters self-register;
   the registry scans the `adapters/` directory, so there is no central list.
   For Anthropic models it's even less: `onramp discover --probe` reads the
   live Models API and auto-registers anything new — **no code at all**.

2. **Empirical capability probes** *(the novel part)*. On onboarding, the
   on-ramp doesn't trust a hand-written config — it **measures the model**:
   JSON-output reliability, deterministic instruction-following rubric,
   tool-call well-formedness, tokens/second, and usable context via
   needle-in-haystack bisection. Results go into a cached, versioned
   **capability manifest**, and every run appends a history snapshot. A hard
   dollar budget caps each onboarding run.

3. **Capability-based routing with staged rollout and failover.** Roles
   ("judge", "trainer", "tool_agent") declare capability *requirements*,
   never model names. Freshly probed models enter as **candidates** —
   routable, but ranked below **stable** models until `onramp promote`
   flips them, so a brand-new model can't hijack production traffic on day
   one. The ranked list *is* the fallback chain: `OnrampClient` retries
   transient failures with backoff, then walks the chain, charging every
   call against a session cost cap and logging to an event stream.

4. **Drift detection + observability.** Re-probing is a scheduled CI job;
   `onramp drift <model>` compares the two latest snapshots and exits
   non-zero when a capability regressed — catching silent model updates
   (same id, new behavior). `onramp serve` gives a live dashboard;
   `onramp export` emits a JSON manifest feed for other dashboards.

```
        new model ships
              │
              ▼
   ┌─────────────────────┐      ┌──────────────────────────┐
   │  adapters/foo_v2.py │─────▶│  Registry (auto-discover) │
   │  (~20 lines)        │      └────────────┬─────────────┘
   └─────────────────────┘                   │ onramp probe foo-v2
                                             ▼
                                ┌──────────────────────────┐
                                │  Probe suite ($-capped)   │
                                │  json · instructions ·    │
                                │  tools · latency · context│
                                └────────────┬─────────────┘
                                             ▼
                                ┌──────────────────────────┐
                                │  Manifest + history       │──▶ drift alerts
                                └────────────┬─────────────┘
                                             ▼
                                ┌──────────────────────────┐
                                │  Router: role -> ranked   │
                                │  candidates = fallback    │
                                └────────────┬─────────────┘
              ┌──────────────────────────────┼──────────────────────────────┐
              ▼                              ▼                              ▼
   ┌────────────────────┐        ┌────────────────────┐        ┌────────────────────┐
   │ Training / GRPO /  │        │ Eval & benchmark   │        │ Judge hierarchies, │
   │ self-learning loops│        │ harnesses          │        │ dashboards, agents │
   └────────────────────┘        └────────────────────┘        └────────────────────┘
        OnrampClient.chat(messages, role="judge")  — no model names anywhere
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the layer-by-layer design.

## Quick start

```sh
pip install -e ".[dev]"

# Works offline — full pipeline against a mock model, no API keys:
python -m pytest tests/ -q
PYTHONPATH=. python3 examples/self_learning_integration.py --mock

# With a real key:
export ANTHROPIC_API_KEY=...
python -m onramp discover --probe           # auto-register NEW models from the
                                            #   Models API and measure them
python -m onramp probe --all --jobs 3       # (re)probe everything in parallel
python -m onramp promote claude-haiku-4-5   # candidate -> stable (serves first)
python -m onramp resolve judge              # best model for a role + chain
python -m onramp drift claude-haiku-4-5     # compare latest two snapshots
python -m onramp serve                      # live dashboard on :8010
```

## Adding a new model (the whole point)

```sh
cp onramp/adapters/_template.py onramp/adapters/my_new_model.py
# fill in: model id, pricing, _complete() — ~20 lines
python -m onramp probe my-new-model --budget 0.50
```

No other file changes. The model is now eligible for every role it qualifies
for, ranked into every fallback chain, and tracked for drift.

## Using it from infrastructure

```python
from onramp import OnrampClient

client = OnrampClient(cost_cap_usd=5.00)          # session budget, enforced
result = client.chat(messages, role="judge")      # routed + auto-failover
print(result.text, result.model_id, result.cost_usd)
```

Roles are defined in `onramp/routing.py` and overridable via a `roles.json`
in your working directory.

## Relationship to self-learning-llm-training

This repo is the foundation layer, and the integration is **live**: the
[self-learning-llm-training](https://github.com/sugeerth/self-learning-llm-training)
loop (Trainer → Evaluator → Judge → MetaJudge → Orchestrator) resolves each
agent's model by role through `onramp_bridge.py`, falling back to its legacy
hard-coded ids whenever no probed model qualifies. Onboard a model, and the
next agent call routes to it — zero changes in the loop's code.

## Roadmap

See [PLAN.md](PLAN.md). Phases 0–3 are implemented, plus drift detection and
scheduled re-probing from Phase 4.

## License

MIT
