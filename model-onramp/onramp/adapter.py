"""The adapter contract: the entire surface a new model must implement.

An adapter subclasses AdapterBase and implements one method, `_complete`.
Timing, token accounting, and the public chat/generate API come for free.
Adapters hold no capability claims beyond pricing — everything else in the
manifest is measured by probes, so a wrong guess in an adapter can't poison
downstream routing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens — the only hand-declared numbers in an adapter."""

    input_per_mtok: float
    output_per_mtok: float

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok
                + output_tokens * self.output_per_mtok) / 1_000_000


@dataclass
class ChatResult:
    """Every model call returns one of these — text plus the accounting
    (tokens, latency, cost) that budget guards and probes rely on."""

    text: str
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float

    def __str__(self) -> str:  # convenient drop-in for str-returning code
        return self.text


def estimate_tokens(text: str) -> int:
    """Rough fallback when a provider doesn't report usage (~4 chars/token)."""
    return max(1, len(text) // 4)


class AdapterBase:
    """Implement `_complete`; everything else is provided."""

    model_id: str
    provider: str
    pricing: Pricing

    def _complete(self, messages: list[dict], max_tokens: int,
                  temperature: float) -> tuple[str, int | None, int | None]:
        """Return (text, input_tokens, output_tokens). Token counts may be
        None when the provider doesn't report usage; they'll be estimated."""
        raise NotImplementedError

    def chat(self, messages: list[dict], *, max_tokens: int = 1024,
             temperature: float = 0.0) -> ChatResult:
        start = time.monotonic()
        text, in_tok, out_tok = self._complete(messages, max_tokens, temperature)
        latency = time.monotonic() - start
        if in_tok is None:
            in_tok = sum(estimate_tokens(str(m.get("content", ""))) for m in messages)
        if out_tok is None:
            out_tok = estimate_tokens(text)
        return ChatResult(
            text=text,
            model_id=self.model_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
            cost_usd=self.pricing.cost_usd(in_tok, out_tok),
        )

    def generate(self, prompt: str, *, max_tokens: int = 1024,
                 temperature: float = 0.0) -> ChatResult:
        return self.chat([{"role": "user", "content": prompt}],
                         max_tokens=max_tokens, temperature=temperature)


@runtime_checkable
class ModelAdapter(Protocol):
    """Structural type used by the registry; AdapterBase satisfies it."""

    model_id: str
    provider: str
    pricing: Pricing

    def generate(self, prompt: str, *, max_tokens: int = 1024,
                 temperature: float = 0.0) -> ChatResult: ...

    def chat(self, messages: list[dict], *, max_tokens: int = 1024,
             temperature: float = 0.0) -> ChatResult: ...
