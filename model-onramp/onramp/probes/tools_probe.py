"""Tool-use probe: given a tool schema, does the model emit well-formed
calls with the right arguments? Probed at the prompt level so it works
identically across providers regardless of native tool-call APIs."""

from __future__ import annotations

import json

from ..adapter import ModelAdapter
from ..budget import CostTracker
from .json_probe import _strip_fences

TOOL_SCHEMA = {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {"city": "string — city name"},
}

TASKS = [
    ("What's the weather in Tokyo right now?", "tokyo"),
    ("Is it raining in Paris?", "paris"),
    ("Tell me the current temperature in Cairo.", "cairo"),
]

PROMPT = (
    "You can call this tool:\n{schema}\n\n"
    "For the user request below, respond with ONLY a JSON object of the "
    'form {{"tool": <tool name>, "arguments": {{...}}}} — no prose.\n\n'
    "User request: {task}"
)


def probe_tool_use(model: ModelAdapter, tracker: CostTracker) -> float:
    successes = 0
    for task, expected_city in TASKS:
        prompt = PROMPT.format(schema=json.dumps(TOOL_SCHEMA), task=task)
        result = model.generate(prompt, max_tokens=150)
        tracker.charge(result)
        try:
            call = json.loads(_strip_fences(result.text))
            city = str(call.get("arguments", {}).get("city", ""))
            if call.get("tool") == "get_weather" and expected_city in city.lower():
                successes += 1
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass
    return successes / len(TASKS)
