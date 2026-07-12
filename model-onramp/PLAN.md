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

## Phase 3 — Integrate self-learning-llm-training ✅

- [x] `onramp_bridge.py` in the parent repo: agents resolve models by ROLE
      (trainer/evaluator/judge/meta_judge/orchestrator) through the router,
      with graceful fallback to the legacy hard-coded ids at every layer.
- [x] `BaseAgent.call()` resolves at call time, so a model onboarded mid-run
      takes over on the very next agent call.
- [x] Onramp dashboard (`onramp serve`) + manifest feed (`onramp export`)
      the training dashboard can poll.
- [ ] Hyperband scheduler reads cost from manifests to allocate budget.

Exit criteria met: onboarding a model (adapter or `onramp discover`) makes
it eligible for every agent role with no code changes in the loop.

## Phase 4 — Continuous re-validation (mostly ✅)

- [x] Manifest history: every probe run appends an immutable snapshot.
- [x] `onramp drift <model>`: compares the two latest snapshots, exits
      non-zero on >10% relative change — catches silent model updates.
- [x] Scheduled re-probing: weekly CI workflow (`.github/workflows/drift.yml`)
      re-probes all models in parallel and fails on drift.
- [x] Manifest feed: `onramp export` publishes registry state as JSON.
- [ ] Alert routing (Slack/email) on drift-workflow failure.

## Phase 6 — Self-learning routing ✅ (new)

The router learns from production, not just probes:

- [x] Live stats store (`onramp/stats.py`): every client call records
      success/failure, cost, and latency per (model, role); consumers add
      quality scores via `client.feedback()`. Laplace-smoothed rates.
- [x] Circuit breaker: 3 consecutive live failures skip a model until a
      cooldown passes (half-open retry after); if every candidate is
      tripped, the chain serves anyway — degraded beats down.
- [x] Bandit exploration: `explore_rate` share of traffic leads with a
      non-first candidate so newcomers earn live evidence.
- [x] Autopilot (`onramp autopilot [--apply]`): promotes candidates with
      enough live calls, high success rate, quality at least matching the
      stable cohort, and known pricing; demotes stable models whose live
      success rate collapses. Every action is evidence-stamped.
- [x] Closed loop with self-learning-llm-training: `BaseAgent.call()`
      reports every outcome; the Judge's verdict scores the Evaluator's
      model and the MetaJudge's audit scores the Judge's model — the
      judging hierarchy literally trains the router.

## Phase 5 — Zero-touch onboarding ✅

- [x] `onramp discover [--probe]`: reads the live Anthropic Models API and
      auto-registers adapters for models not yet known — a brand-new model
      onboards with **zero files written**. Pricing resolves by longest
      prefix from `discovery.KNOWN_PRICING`; unknown-price models are
      flagged and blocked from promotion.
- [x] Staged rollout: probed models start as `candidate` (routable, ranked
      below `stable`); `onramp promote/demote/retire` manage the lifecycle,
      so a new model can't take production traffic until a human flips it.

## Non-goals

- Not another LLM gateway/proxy (no request routing service, no auth layer).
- Not a benchmark leaderboard — probes measure *plumbing* capabilities needed
  by infrastructure, not general intelligence.
- No heavyweight plugin system — an adapter is a plain Python file.
