"""Generic adapter for any OpenAI-compatible chat endpoint (vLLM, Ollama,
llama.cpp server, hosted providers). Uses stdlib urllib — no extra dependency.

Registers only when OPENAI_COMPAT_MODEL is set, since the model id, endpoint,
and pricing are deployment-specific:

    export OPENAI_COMPAT_MODEL=llama-3.3-70b
    export OPENAI_COMPAT_BASE_URL=http://localhost:8000/v1
    export OPENAI_COMPAT_API_KEY=...            # optional
    export OPENAI_COMPAT_PRICING=0.6,0.6        # optional $in,$out per MTok
"""

from __future__ import annotations

import json
import os
import urllib.request

from ..adapter import AdapterBase, Pricing
from ..registry import register


class OpenAICompatAdapter(AdapterBase):
    provider = "openai-compatible"
    base_url = "http://localhost:8000/v1"
    api_key = ""

    def _complete(self, messages, max_tokens, temperature):
        payload = json.dumps({
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read())
        usage = data.get("usage", {})
        return (
            data["choices"][0]["message"]["content"],
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )


_model = os.environ.get("OPENAI_COMPAT_MODEL")
if _model:
    _in, _, _out = os.environ.get("OPENAI_COMPAT_PRICING", "0,0").partition(",")
    register(type("OpenAICompat", (OpenAICompatAdapter,), {
        "model_id": _model,
        "base_url": os.environ.get("OPENAI_COMPAT_BASE_URL",
                                   OpenAICompatAdapter.base_url).rstrip("/"),
        "api_key": os.environ.get("OPENAI_COMPAT_API_KEY", ""),
        "pricing": Pricing(float(_in or 0), float(_out or 0)),
    }))
