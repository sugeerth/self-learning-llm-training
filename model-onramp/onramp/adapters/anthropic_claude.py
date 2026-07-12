"""Anthropic Claude adapters — the reference implementation.

One base class, one registered adapter per served model. Pricing is USD per
million tokens from Anthropic's published rates."""

from __future__ import annotations

import os

from ..adapter import AdapterBase, Pricing
from ..registry import register

CLAUDE_MODELS = [
    # (model_id, input $/MTok, output $/MTok)
    ("claude-opus-4-8", 5.00, 25.00),
    ("claude-sonnet-5", 3.00, 15.00),
    ("claude-haiku-4-5", 1.00, 5.00),
]


class ClaudeAdapter(AdapterBase):
    provider = "anthropic"

    def __init__(self) -> None:
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic  # lazy: registry discovery must not require the SDK

            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self._client

    def _complete(self, messages, max_tokens, temperature):
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return text, response.usage.input_tokens, response.usage.output_tokens


for _model_id, _in, _out in CLAUDE_MODELS:
    register(type(
        f"Claude_{_model_id.replace('-', '_')}",
        (ClaudeAdapter,),
        {"model_id": _model_id, "pricing": Pricing(_in, _out)},
    ))
