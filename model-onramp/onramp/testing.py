"""Mock adapters for tests, demos, and CI — no API key, no network.

perfect_responder answers every probe correctly; broken_responder fails them
all. Between them you can exercise the full onboarding pipeline offline.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from .adapter import AdapterBase, Pricing, estimate_tokens


def perfect_responder(prompt: str) -> str:
    # Context probe: retrieve the needle
    if "secret code" in prompt:
        match = re.search(r"XK-\d+-OMEGA", prompt)
        return match.group(0) if match else "unknown"
    # Tool probe
    if '"tool"' in prompt and "User request:" in prompt:
        for city in ("Tokyo", "Paris", "Cairo"):
            if city in prompt:
                return json.dumps({"tool": "get_weather",
                                   "arguments": {"city": city}})
        return json.dumps({"tool": "get_weather", "arguments": {"city": "?"}})
    # JSON probe: echo the requested shape with plausible values
    if "Return ONLY a JSON object" in prompt:
        result: dict = {}
        for key, kind in re.findall(r'"(\w+)": <(\w+)>', prompt):
            result[key] = {"string": "mock", "integer": 1, "boolean": True}[kind]
        for key in re.findall(r'"(\w+)": \[<integer>', prompt):
            result[key] = [2, 3, 5, 7]
        return json.dumps(result or {"ok": True})
    # Instruction probe rubric
    if "exactly three words" in prompt:
        return "Vast blue deep"
    if "ALL CAPS" in prompt:
        return "PARIS"
    if "ending your reply with the word BANANA" in prompt:
        return "Apple BANANA"
    if "single word: yes" in prompt:
        return "yes"
    if "four seasons" in prompt:
        return "1. Spring\n2. Summer\n3. Autumn\n4. Winter"
    # Latency probe / everything else
    return ", ".join(str(i) for i in range(1, 51))


def broken_responder(prompt: str) -> str:
    return "I'm sorry, I can't help with that request right now!!"


class MockAdapter(AdapterBase):
    """Configurable fake model. Not auto-registered — tests call
    register(make_mock(...)) explicitly."""

    provider = "mock"

    def __init__(self) -> None:
        self.calls = 0

    def _complete(self, messages, max_tokens, temperature):
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        prompt = str(messages[-1].get("content", ""))
        if self.max_context_tokens is not None:
            total = sum(estimate_tokens(str(m.get("content", ""))) for m in messages)
            if total > self.max_context_tokens:
                return "The document was too long; I lost track.", None, None
        return self.responder(prompt), None, None


def make_mock(model_id: str, *, pricing: Pricing = Pricing(1.0, 2.0),
              responder: Callable[[str], str] = perfect_responder,
              fail_with: Exception | None = None,
              max_context_tokens: int | None = None) -> type:
    """Build a MockAdapter subclass suitable for register()."""
    return type(f"Mock_{model_id}", (MockAdapter,), {
        "model_id": model_id,
        "pricing": pricing,
        "responder": staticmethod(responder),
        "fail_with": fail_with,
        "max_context_tokens": max_context_tokens,
    })
