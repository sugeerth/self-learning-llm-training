# model-onramp

> Models are plugins. Infrastructure is permanent.

**model-onramp** is a model-onboarding layer for LLM infrastructure. When a new
model ships — a new Claude, a new open-weights release, a fine-tune of your own —
you add **one small adapter file**. Everything built on top (training loops,
evals, judges, benchmarks, dashboards) picks it up automatically, with zero
changes to the infrastructure itself.

## The idea

Most LLM stacks hard-code their models: model names in config files, provider
SDKs called directly from training code, capability assumptions (context
length, tool use, JSON reliability) baked into prompts and schedulers. Every
new model release means a sweep through the whole codebase.

model-onramp inverts that:

1. **One-file adapters.** A new model enters the system as a single file
   implementing the `ModelAdapter` protocol (`generate`, `chat`, metadata).
   Copy `_template.py`, fill in ~30 lines, done.

2. **Auto-discovering registry.** Adapters register themselves via a decorator.
   The registry scans the `adapters/` directory — no central list to edit.

3. **Empirical capability probes** *(the novel part)*. On registration, the
   on-ramp doesn't trust a hand-written config — it **probes the model** and
   generates a capability manifest: usable context length, tool-use support,
   JSON-output reliability, instruction-following score, latency, and cost per
   token. The manifest is cached and versioned.

4. **Infrastructure consumes manifests, not model names.** Downstream systems
   query the registry by *capability* ("give me all models with reliable JSON
   output and ≥100k context") instead of by name. A Hyperband scheduler, a
   judge hierarchy, or a self-learning loop configures itself from the
   manifest — so a model released tomorrow works with infrastructure written
   today.

```
        new model ships
              │
              ▼
   ┌─────────────────────┐      ┌──────────────────────────┐
   │  adapters/foo_v2.py │─────▶│  Registry (auto-discover) │
   │  (~30 lines)        │      └────────────┬─────────────┘
   └─────────────────────┘                   │ on register
                                             ▼
                                ┌──────────────────────────┐
                                │  Capability probes        │
                                │  context · tools · JSON   │
                                │  latency · cost           │
                                └────────────┬─────────────┘
                                             │ emits
                                             ▼
                                ┌──────────────────────────┐
                                │  Capability manifest      │
                                │  (cached, versioned)      │
                                └────────────┬─────────────┘
              ┌──────────────────────────────┼──────────────────────────────┐
              ▼                              ▼                              ▼
   ┌────────────────────┐        ┌────────────────────┐        ┌────────────────────┐
   │ Training / GRPO /  │        │ Eval & benchmark   │        │ Judge hierarchies, │
   │ self-learning loops│        │ harnesses          │        │ dashboards, agents │
   └────────────────────┘        └────────────────────┘        └────────────────────┘
```

## Quick start

```sh
pip install -e .
export ANTHROPIC_API_KEY=...

# List discovered models and their manifests
python -m onramp list

# Onboard (probe) a newly added adapter
python -m onramp probe claude-fable-5

# Query by capability from your own code
python examples/query_by_capability.py
```

## Adding a new model (the whole point)

```sh
cp onramp/adapters/_template.py onramp/adapters/my_new_model.py
# fill in: model id, provider call, pricing — ~30 lines
python -m onramp probe my-new-model   # generates its capability manifest
```

No other file changes. Every consumer of the registry now sees the model.

## Relationship to self-learning-llm-training

This repo is the foundation layer. The
[self-learning-llm-training](https://github.com/sugeerth/self-learning-llm-training)
loop (Trainer → Evaluator → Judge → MetaJudge) becomes a *consumer*: instead of
hard-coding an Anthropic model, it asks the registry for models matching the
capabilities each agent role needs. See
`examples/self_learning_integration.py`.

## Roadmap

See [PLAN.md](PLAN.md) for the phased plan.

## License

MIT
