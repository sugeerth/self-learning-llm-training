# model-onramp — Phased Plan

Goal: infrastructure that never needs to change when a new model is released.
A new model = one adapter file + an automatic probe run. Everything downstream
(training, eval, judging, dashboards) works on top of the registry.

## Phase 0 — Contract ✅

- [x] `AdapterBase` / `ModelAdapter`: implement `_complete()`, get timing,
      token accounting, and per-call cost (`ChatResult`) for free.
- [x] Self-registering adapter pattern (`@register` + directory scan) with
      `unregister` for tests/hot-swap.
- [x] Capability manifest schema (dataclass, JSON-serializable, versioned).
- [x] Adapters: Anthropic Claude (Opus 4.8 / Sonnet 5 / Haiku 4.5), generic
      OpenAI-compatible endpoint (env-configured), `_template.py`.
- [x] Mock adapters (`onramp/testing.py`) so the full pipeline runs offline.

Exit criteria met: a second adapter is added by copying the template and
touching nothing else.

## Phase 1 — Empirical capability probes ✅

- [x] JSON probe: 5 varied schemas → parse-success rate.
- [x] Instruction probe: deterministic 5-item rubric (no judge model needed).
- [x] Tool-use probe: schema → well-formed-call rate with argument checks.
- [x] Latency probe: measured output tokens/second.
- [x] Context probe: needle-in-haystack at doubling sizes (1k → 128k),
      reports largest passing size; stops early on budget or provider limits.
- [x] Manifest cache keyed by probe-suite version; stale manifests re-probe.
- [x] Probe budget guard: hard USD cap per onboarding run
      (`onramp probe <id> --budget 0.50`); partial manifests persist.
- [x] Event stream (`.onramp/events.jsonl`) for dashboard consumption.

## Phase 2 — Capability-based routing ✅

- [x] `registry.find(**needs)` query API (min-threshold for numbers, exact
      for booleans; unprobed models never qualify).
- [x] Role profiles (judge / meta_judge / trainer / drafter / tool_agent),
      overridable via `roles.json`; rank by cost or speed.
- [x] Fallback chains: `Router.candidates(role)` is the ordered chain;
      `OnrampClient` walks it on provider failure with per-session cost caps.

## Phase 3 — Integrate self-learning-llm-training

- [ ] Replace hard-coded model ids in the Trainer/Evaluator/Judge/MetaJudge
      agents with `OnrampClient(role=...)` calls
      (see `examples/self_learning_integration.py` for the pattern).
- [ ] Hyperband scheduler reads cost from manifests to allocate budget.
- [ ] Dashboard panel: registered models, manifest diffs, event stream tail.

Exit criteria: dropping in an adapter for a brand-new model makes it eligible
for every agent role in the self-learning loop with no code changes there.

## Phase 4 — Continuous re-validation (drift detection ✅, scheduling ☐)

- [x] Manifest history: every probe run appends an immutable snapshot.
- [x] `onramp drift <model>`: compares the two latest snapshots, exits
      non-zero on >10% relative change — catches silent model updates.
- [ ] Scheduled re-probing (cron / CI job) + alerting on drift.
- [ ] Publish manifests as a static JSON feed the dashboard consumes.

## Non-goals

- Not another LLM gateway/proxy (no request routing service, no auth layer).
- Not a benchmark leaderboard — probes measure *plumbing* capabilities needed
  by infrastructure, not general intelligence.
- No heavyweight plugin system — an adapter is a plain Python file.
