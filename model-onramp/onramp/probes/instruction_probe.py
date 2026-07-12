"""Instruction probe: a fixed rubric of constraints that are checkable
deterministically — no judge model needed, so the score is reproducible."""

from __future__ import annotations

from ..adapter import ModelAdapter
from ..budget import CostTracker


def _exactly_three_words(text: str) -> bool:
    return len(text.strip().split()) == 3

def _all_caps(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped == stripped.upper()

def _ends_with_banana(text: str) -> bool:
    return text.strip().rstrip(".!").upper().endswith("BANANA")

def _single_word_yes(text: str) -> bool:
    return text.strip().rstrip(".!").lower() == "yes"

def _numbered_list_of_four(text: str) -> bool:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return len(lines) == 4 and all(
        l.strip().startswith(f"{i}.") for i, l in enumerate(lines, 1))


CHECKS = [
    ("Describe the ocean in exactly three words.", _exactly_three_words),
    ("Reply in ALL CAPS: what is the capital of France?", _all_caps),
    ("Name any fruit, ending your reply with the word BANANA.", _ends_with_banana),
    ("Is water wet? Answer with the single word: yes", _single_word_yes),
    ("List exactly four seasons as a numbered list (1. ... to 4. ...), "
     "one per line, nothing else.", _numbered_list_of_four),
]


def probe_instruction_following(model: ModelAdapter,
                                tracker: CostTracker) -> float:
    passed = 0
    for prompt, check in CHECKS:
        result = model.generate(prompt, max_tokens=100)
        tracker.charge(result)
        if check(result.text):
            passed += 1
    return passed / len(CHECKS)
