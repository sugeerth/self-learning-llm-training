"""JSON probe: can the model reliably emit machine-parseable output?"""

from __future__ import annotations

import json

from ..adapter import ModelAdapter
from ..budget import CostTracker

# Varied schemas so a model can't luck into one memorized answer.
PROMPTS = [
    ('Return ONLY a JSON object of the form {"city": <string>, '
     '"population": <integer>} for the largest city in Japan.', "city"),
    ('Return ONLY a JSON object of the form {"language": <string>, '
     '"typed": <boolean>} describing Python.', "language"),
    ('Return ONLY a JSON object of the form {"primes": [<integer>, ...]} '
     'with the first four prime numbers.', "primes"),
    ('Return ONLY a JSON object of the form {"color": <string>, '
     '"hex": <string>} for the color of a clear daytime sky.', "color"),
    ('Return ONLY a JSON object of the form {"planet": <string>, '
     '"moons": <integer>} for Mars.', "planet"),
]
SUFFIX = " No prose, no code fences, no explanation."


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def probe_json_reliability(model: ModelAdapter, tracker: CostTracker) -> float:
    successes = 0
    for prompt, required_key in PROMPTS:
        result = model.generate(prompt + SUFFIX, max_tokens=120)
        tracker.charge(result)
        try:
            parsed = json.loads(_strip_fences(result.text))
            if isinstance(parsed, dict) and required_key in parsed:
                successes += 1
        except (json.JSONDecodeError, ValueError):
            pass
    return successes / len(PROMPTS)
