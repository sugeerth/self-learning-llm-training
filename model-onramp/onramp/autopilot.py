"""Autopilot: lifecycle decisions from live evidence, not vibes.

Probes decide whether a model is *eligible* (candidate). Live traffic —
earned via the client's exploration share — decides whether it's *trusted*
(stable). Autopilot closes that loop:

  PROMOTE  candidate -> stable   when it has enough live calls, a high
                                 success rate, a mean quality score (when
                                 scores exist) at least matching the best
                                 stable model's, and known pricing.
  DEMOTE   stable -> candidate   when its live success rate collapses —
                                 the durable response to what the circuit
                                 breaker handles transiently.

Run `onramp autopilot` (dry-run) on a schedule; `--apply` executes.
Every action is evidence-stamped in the event stream.
"""

from __future__ import annotations

from dataclasses import dataclass

from .capabilities import CapabilityManifest
from .events import emit
from .registry import Registry, get_registry
from .stats import StatsStore, get_stats


@dataclass
class Action:
    model_id: str
    action: str        # "promote" | "demote"
    reason: str


def evaluate(*, registry: Registry | None = None,
             stats: StatsStore | None = None,
             min_calls: int = 25,
             promote_success_rate: float = 0.95,
             demote_success_rate: float = 0.70,
             score_margin: float = 0.05) -> list[Action]:
    """Return the lifecycle actions the live evidence supports."""
    registry = registry or get_registry()
    stats = stats or get_stats()
    actions: list[Action] = []

    manifests = {m: registry.manifest(m) for m in registry.model_ids()}
    stable_scores = [s for s in (stats.mean_score(m) for m, mf in
                                 manifests.items()
                                 if mf and mf.status == "stable")
                     if s is not None]
    score_bar = max(stable_scores) - score_margin if stable_scores else None

    for model_id, manifest in manifests.items():
        if manifest is None:
            continue
        calls = stats.calls(model_id)
        if calls < min_calls:
            continue
        rate = stats.success_rate(model_id)

        if manifest.status == "candidate":
            if manifest.notes.get("pricing_unknown"):
                continue  # never auto-promote an unpriced model
            if rate < promote_success_rate:
                continue
            score = stats.mean_score(model_id)
            if score_bar is not None and score is not None and score < score_bar:
                continue  # quality measurably below the stable cohort
            actions.append(Action(
                model_id, "promote",
                f"{calls} live calls, success={rate:.3f}"
                + (f", score={score:.3f}" if score is not None else "")))

        elif manifest.status == "stable" and rate < demote_success_rate:
            actions.append(Action(
                model_id, "demote",
                f"{calls} live calls, success={rate:.3f} < "
                f"{demote_success_rate}"))

    return actions


def apply(actions: list[Action]) -> None:
    for act in actions:
        manifest = CapabilityManifest.load(act.model_id)
        if manifest is None:
            continue
        manifest.set_status("stable" if act.action == "promote" else "candidate")
        emit("autopilot", model_id=act.model_id, action=act.action,
             reason=act.reason)
