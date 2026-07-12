"""Latency probe: measured output tokens/second on a mid-size completion."""

from __future__ import annotations

from ..adapter import ModelAdapter
from ..budget import CostTracker


def probe_latency(model: ModelAdapter, tracker: CostTracker) -> float:
    result = model.generate("Count from 1 to 50, comma-separated.",
                            max_tokens=300)
    tracker.charge(result)
    if result.latency_s <= 0:
        return 0.0
    return round(result.output_tokens / result.latency_s, 1)
