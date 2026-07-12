"""Context probe: needle-in-a-haystack retrieval at doubling document sizes.
Reports the largest size (in approx tokens) at which retrieval still works —
the *usable* context, which is often smaller than the advertised window."""

from __future__ import annotations

from ..adapter import ModelAdapter, estimate_tokens
from ..budget import CostTracker

FILLER_SENTENCE = (
    "The archive room holds ledgers, maps, weather logs, and shipping "
    "records from many uneventful years. ")
NEEDLE = "The secret code is {code}."
QUESTION = "\n\nWhat is the secret code mentioned in the document? Reply with only the code."

SIZES_TOKENS = (1_000, 4_000, 16_000, 64_000, 128_000)


def _build_document(target_tokens: int, code: str) -> str:
    filler_tokens = estimate_tokens(FILLER_SENTENCE)
    n = max(1, target_tokens // filler_tokens)
    sentences = [FILLER_SENTENCE] * n
    sentences.insert(n // 2, NEEDLE.format(code=code) + " ")
    return "".join(sentences)


def probe_usable_context(model: ModelAdapter, tracker: CostTracker, *,
                         max_tokens: int = 64_000) -> int:
    largest_passing = 0
    for i, size in enumerate(s for s in SIZES_TOKENS if s <= max_tokens):
        code = f"XK-{41 + i}-OMEGA"
        prompt = _build_document(size, code) + QUESTION
        estimated_cost = model.pricing.cost_usd(estimate_tokens(prompt), 50)
        if tracker.would_exceed(estimated_cost):
            break  # stop escalating rather than blowing the budget
        try:
            result = model.generate(prompt, max_tokens=50)
        except Exception:
            break  # provider rejected the length — previous size stands
        tracker.charge(result)
        if code not in result.text:
            break
        largest_passing = size
    return largest_passing
