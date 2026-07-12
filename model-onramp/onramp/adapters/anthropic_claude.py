"""Anthropic Claude adapter — the reference adapter for the on-ramp."""

import os

from ..adapter import Pricing
from ..registry import register


@register
class ClaudeSonnetAdapter:
    model_id = "claude-sonnet-5"
    provider = "anthropic"
    pricing = Pricing(input_per_mtok=3.00, output_per_mtok=15.00)

    def __init__(self) -> None:
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self._client

    def generate(self, prompt: str, *, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        return self.chat([{"role": "user", "content": prompt}],
                         max_tokens=max_tokens, temperature=temperature)

    def chat(self, messages: list, *, max_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        return response.content[0].text
