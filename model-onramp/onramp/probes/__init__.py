"""Empirical capability probes: on onboarding, a model is *measured*, not
trusted. Each probe exercises the adapter through its public API and writes
its result into the CapabilityManifest. Every call is charged to a
CostTracker so onboarding can never exceed its dollar budget."""

from __future__ import annotations

from ..adapter import ModelAdapter
from ..budget import BudgetExceededError, CostTracker
from ..capabilities import CapabilityManifest
from ..events import emit
from .context_probe import probe_usable_context
from .instruction_probe import probe_instruction_following
from .json_probe import probe_json_reliability
from .latency_probe import probe_latency
from .tools_probe import probe_tool_use

__all__ = ["run_probes", "BudgetExceededError"]


def run_probes(model: ModelAdapter, *, budget_usd: float = 1.0,
               skip_context: bool = False,
               max_context_tokens: int = 64_000) -> CapabilityManifest:
    """Run the full suite and persist the manifest (+ history snapshot).

    Probes run cheapest-first so a tight budget still yields a usable
    partial manifest; a BudgetExceededError mid-suite saves what was
    measured so far before re-raising."""
    tracker = CostTracker(cap_usd=budget_usd)
    manifest = CapabilityManifest(
        model_id=model.model_id,
        provider=model.provider,
        input_per_mtok=model.pricing.input_per_mtok,
        output_per_mtok=model.pricing.output_per_mtok,
    )
    emit("probe_start", model_id=model.model_id, budget_usd=budget_usd)
    try:
        manifest.json_reliability = probe_json_reliability(model, tracker)
        manifest.instruction_score = probe_instruction_following(model, tracker)
        manifest.tool_use_reliability = probe_tool_use(model, tracker)
        manifest.tokens_per_second = probe_latency(model, tracker)
        if not skip_context:
            manifest.usable_context_tokens = probe_usable_context(
                model, tracker, max_tokens=max_context_tokens)
    except BudgetExceededError as err:
        manifest.notes["budget_exceeded"] = str(err)
        raise
    finally:
        manifest.probe_cost_usd = round(tracker.spent_usd, 6)
        manifest.save()
        emit("probe_done", model_id=model.model_id,
             cost_usd=manifest.probe_cost_usd,
             json=manifest.json_reliability,
             tools=manifest.tool_use_reliability,
             instructions=manifest.instruction_score,
             context=manifest.usable_context_tokens,
             tps=manifest.tokens_per_second)
    return manifest
