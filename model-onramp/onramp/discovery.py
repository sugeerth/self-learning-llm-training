"""Zero-adapter onboarding: discover new Anthropic models from the live
Models API and register adapters for them automatically.

This closes the loop on "the infrastructure works whenever new models are
added": when Anthropic ships a model tomorrow, `onramp discover --probe`
registers it, measures it, and makes it routable — no code written at all.

Pricing cannot be fetched from the Models API, so discovered models get
pricing from the longest-prefix match in KNOWN_PRICING, else $0 with a
`pricing_unknown` note. Unknown-price models still onboard, but stay
"candidate" until a human confirms pricing and promotes them — the
stable-first router keeps them out of the serving path until then.
"""

from __future__ import annotations

from typing import Callable, Iterable

from .adapter import Pricing
from .capabilities import CapabilityManifest
from .events import emit
from .registry import get_registry, register

# Longest-prefix pricing map (USD per MTok). Extend as families ship.
KNOWN_PRICING: dict[str, Pricing] = {
    "claude-fable-5": Pricing(10.00, 50.00),
    "claude-opus-4": Pricing(5.00, 25.00),
    "claude-sonnet-5": Pricing(3.00, 15.00),
    "claude-sonnet-4": Pricing(3.00, 15.00),
    "claude-haiku-4": Pricing(1.00, 5.00),
}


def list_anthropic_models() -> list[str]:
    """Live model ids from the Anthropic Models API."""
    import anthropic

    client = anthropic.Anthropic()
    return [model.id for model in client.models.list()]


def _pricing_for(model_id: str) -> Pricing | None:
    matches = [p for p in KNOWN_PRICING if model_id.startswith(p)]
    if not matches:
        return None
    return KNOWN_PRICING[max(matches, key=len)]


def discover(lister: Callable[[], Iterable[str]] = list_anthropic_models,
             ) -> list[str]:
    """Register an adapter for every listed model not already known.
    Returns the newly registered model ids."""
    from .adapters.anthropic_claude import ClaudeAdapter

    registry = get_registry()
    new_ids = []
    for model_id in lister():
        if model_id in registry:
            continue
        pricing = _pricing_for(model_id)
        register(type(
            f"Discovered_{model_id.replace('-', '_').replace('.', '_')}",
            (ClaudeAdapter,),
            {"model_id": model_id, "pricing": pricing or Pricing(0.0, 0.0)},
        ))
        if pricing is None:
            # flag it so nobody promotes an unpriced model by accident
            manifest = (CapabilityManifest.load(model_id)
                        or CapabilityManifest(model_id=model_id))
            manifest.notes["pricing_unknown"] = True
            manifest.save(snapshot=False)
        new_ids.append(model_id)
        emit("discovered", model_id=model_id,
             pricing_known=pricing is not None)
    return new_ids
