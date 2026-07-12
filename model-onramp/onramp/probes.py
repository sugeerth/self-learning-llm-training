"""Empirical capability probes (Phase 1).

On onboarding, a model is *measured*, not trusted: each probe exercises the
adapter and writes its result into the CapabilityManifest. This scaffold
implements the cheap probes; context bisection and judge-scored instruction
probes are Phase 1 work (see PLAN.md).
"""

from __future__ import annotations

import json
import time

from .adapter import ModelAdapter
from .capabilities import CapabilityManifest

JSON_PROBE_ROUNDS = 5


def probe_json_reliability(model: ModelAdapter) -> float:
    """Ask for strict JSON N times; return the parse-success rate."""
    prompt = (
        'Return ONLY a JSON object of the form {"city": <string>, '
        '"population": <integer>} for the largest city in Japan. '
        "No prose, no code fences."
    )
    successes = 0
    for _ in range(JSON_PROBE_ROUNDS):
        try:
            reply = model.generate(prompt, max_tokens=100)
            parsed = json.loads(reply.strip())
            if isinstance(parsed, dict) and "city" in parsed:
                successes += 1
        except Exception:
            pass
    return successes / JSON_PROBE_ROUNDS


def probe_latency(model: ModelAdapter) -> float:
    """Rough tokens/sec on a single mid-size completion."""
    start = time.monotonic()
    reply = model.generate("Count from 1 to 50, comma-separated.", max_tokens=300)
    elapsed = time.monotonic() - start
    approx_tokens = max(1, len(reply) // 4)
    return approx_tokens / elapsed if elapsed > 0 else 0.0


def run_probes(model: ModelAdapter) -> CapabilityManifest:
    manifest = CapabilityManifest(
        model_id=model.model_id,
        input_per_mtok=model.pricing.input_per_mtok,
        output_per_mtok=model.pricing.output_per_mtok,
    )
    manifest.json_reliability = probe_json_reliability(model)
    manifest.tokens_per_second = round(probe_latency(model), 1)
    manifest.save()
    return manifest
