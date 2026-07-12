"""The adapter contract: the entire surface a new model must implement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens. The only hand-declared numbers in an adapter;
    everything else in the manifest is measured by probes."""

    input_per_mtok: float
    output_per_mtok: float


@runtime_checkable
class ModelAdapter(Protocol):
    """One file, one class, ~30 lines: that's a model onboarded.

    Adapters hold no capability claims beyond pricing — context length,
    JSON reliability, tool use, etc. are measured by the probe suite and
    stored in the CapabilityManifest, so a wrong guess in an adapter can't
    poison downstream routing.
    """

    model_id: str
    provider: str
    pricing: Pricing

    def generate(self, prompt: str, *, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        """Single-turn completion."""
        ...

    def chat(self, messages: list[dict], *, max_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        """Multi-turn chat. `messages` is [{"role": ..., "content": ...}]."""
        ...
