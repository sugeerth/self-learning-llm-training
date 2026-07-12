# model-onramp — Architecture

**Models are plugins; infrastructure is permanent.** The system has four
layers; each depends only on the one below it, and no layer above the
adapters ever names a model.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. CONSUMERS         training loops · evals · judges · dashboards       │
│    OnrampClient.chat(messages, role="judge")                            │
│      -> routing + failover + cost cap + events, zero model names        │
├─────────────────────────────────────────────────────────────────────────┤
│ 3. ROUTING           Router / RoleProfile        (onramp/routing.py)    │
│    roles declare capability NEEDS ("judge needs json>=0.9");            │
│    candidates() ranks eligible models by cost or speed — the ranked     │
│    list IS the failover chain                                           │
├─────────────────────────────────────────────────────────────────────────┤
│ 2. MEASUREMENT       probe suite + manifests     (onramp/probes/,       │
│    empirical, budget-capped, versioned:           capabilities.py)      │
│      json · instructions · tools · latency · context bisection          │
│    every save -> history snapshot -> drift detection                    │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. ADAPTERS          one file per model          (onramp/adapters/)     │
│    AdapterBase: implement _complete(), get timing/tokens/cost free;     │
│    @register + directory scan -> no central list to edit                │
└─────────────────────────────────────────────────────────────────────────┘

  cross-cutting: budget.py (hard $ caps) · events.py (JSONL stream)
                 paths.py (all state under $ONRAMP_HOME, default ./.onramp)
```

## The onboarding flow

```
cp adapters/_template.py adapters/foo_v2.py      # ~20 lines
python -m onramp probe foo-v2                    # measured, not trusted
# -> .onramp/manifests/foo-v2.json  (+ history snapshot)
# -> foo-v2 is now eligible for every role it qualifies for
```

No other file changes. `Router.resolve("judge")` may now return `foo-v2` if
it probes better/cheaper than the incumbents — infrastructure written before
foo-v2 existed routes to it automatically.

## Key design decisions

- **Manifests are measured, not declared.** Adapters may only declare
  pricing. Everything routing depends on (JSON reliability, usable context,
  tool use, instruction following, speed) comes from probes, so a wrong guess
  in an adapter can't poison routing.
- **Deterministic probes.** The instruction rubric and JSON/tool checks are
  verifiable in code — no judge model — so scores are reproducible and cheap.
- **Budget guards everywhere.** Probing and client sessions charge every call
  to a `CostTracker`; exceeding the cap raises (probes still persist the
  partial manifest).
- **Probe-suite versioning.** Manifests record `probe_suite_version`; bumping
  it invalidates all caches, forcing re-onboarding under the new suite.
- **Drift detection.** Every probe run appends to per-model history;
  `onramp drift <id>` compares the two latest snapshots and exits non-zero on
  regression — run it on a schedule to catch silent model updates.
- **Offline-testable.** `onramp/testing.py` provides mock adapters that pass
  (or fail) every probe, so the entire pipeline runs in CI with no keys.

## State layout (`$ONRAMP_HOME`, default `./.onramp`)

```
.onramp/
  manifests/<model_id>.json          current manifest (cache)
  history/<model_id>/<ts>.json       immutable snapshots (drift source)
  events.jsonl                       append-only event stream
```
