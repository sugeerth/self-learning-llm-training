# model-onramp — Phased Plan

Goal: infrastructure that never needs to change when a new model is released.
A new model = one adapter file + an automatic probe run. Everything downstream
(training, eval, judging, dashboards) works on top of the registry.

## Phase 0 — Contract (this scaffold)

- [x] `ModelAdapter` protocol: the minimal surface every model must expose
      (`generate`, `chat`, `model_id`, `pricing`).
- [x] Self-registering adapter pattern (`@register` decorator + directory scan).
- [x] Capability manifest schema (dataclass, JSON-serializable, versioned).
- [x] One real adapter (Anthropic Claude) + `_template.py` for new models.
- [x] CLI skeleton: `python -m onramp list | probe <model>`.

Exit criteria: a second adapter can be added by copying the template and
touching nothing else.

## Phase 1 — Empirical capability probes

The differentiator: manifests are **measured, not declared**.

- [ ] Probe suite, each probe returning a scored result:
  - context probe: bisection on prompt length until failure → usable context
  - JSON probe: N structured-output requests → parse-success rate
  - tool-use probe: does the model emit well-formed tool calls?
  - instruction probe: small fixed rubric, scored by a judge model
  - latency/cost probe: tokens/sec and $/1k tokens measured live
- [ ] Manifest cache keyed by (model_id, probe-suite version); re-probe only
      when the suite changes or on `--force`.
- [ ] Probe budget guard: hard cap on $ spent per onboarding run.

## Phase 2 — Capability-based routing

- [ ] `registry.find(needs=...)` query API: filter and rank models by manifest
      fields (e.g. `json_reliability >= 0.95, context >= 100_000`, sort by cost).
- [ ] Role profiles: named capability requirements ("judge", "drafter",
      "long-context-reader") defined once, resolved to concrete models at runtime.
- [ ] Fallback chains: ordered candidate lists with automatic failover.

## Phase 3 — Integrate self-learning-llm-training

- [ ] Replace hard-coded model ids in the Trainer/Evaluator/Judge/MetaJudge
      agents with role-profile lookups against the registry.
- [ ] Hyperband scheduler reads cost from manifests to allocate budget.
- [ ] Dashboard panel: registered models, manifest diffs over time.

Exit criteria: dropping in an adapter for a brand-new model makes it eligible
for every agent role in the self-learning loop with no code changes there.

## Phase 4 — Continuous re-validation

- [ ] Scheduled re-probing: detect silent model updates (same id, new behavior)
      by manifest drift.
- [ ] Manifest history + diff alerts ("json_reliability dropped 0.98 → 0.85").
- [ ] Optional: publish manifests as a static JSON feed the dashboard consumes.

## Non-goals

- Not another LLM gateway/proxy (no request routing service, no auth layer).
- Not a benchmark leaderboard — probes measure *plumbing* capabilities needed
  by infrastructure, not general intelligence.
- No heavyweight plugin system — an adapter is a plain Python file.
