"""How self-learning-llm-training plugs into the on-ramp (Phase 3).

Instead of hard-coding a model id per agent, each agent role resolves a
model at runtime through OnrampClient — which also gives it failover and
session cost caps for free. When a new model is onboarded (one adapter file
+ a probe run), it becomes eligible for every role it qualifies for with
zero changes in the self-learning repo.

Run offline (no API keys) with a mock model:

    PYTHONPATH=. python3 examples/self_learning_integration.py --mock
"""

import sys

from onramp import OnrampClient, register
from onramp.routing import NoEligibleModelError


def main() -> None:
    if "--mock" in sys.argv:
        from onramp.probes import run_probes
        from onramp.registry import get_registry
        from onramp.testing import make_mock

        register(make_mock("demo-model"))
        run_probes(get_registry().get("demo-model"), budget_usd=1.0,
                   max_context_tokens=16_000)

    client = OnrampClient(cost_cap_usd=5.00)  # whole session capped at $5

    for role in ("judge", "meta_judge", "trainer", "drafter", "tool_agent"):
        try:
            candidates = client.router.candidates(role)
            print(f"{role:<12} -> {candidates[0]:<20} (fallback chain: {candidates})")
        except (NoEligibleModelError, IndexError):
            print(f"{role:<12} -> no eligible model; probe one first")

    # An agent call: no model name anywhere.
    try:
        verdict = client.generate(
            'Return ONLY a JSON object of the form {"verdict": <string>, '
            '"confidence": <integer>} judging the answer "42".',
            role="judge", max_tokens=100)
        print(f"\njudge said: {verdict.text!r}")
        print(f"served by {verdict.model_id}, cost ${verdict.cost_usd:.6f}, "
              f"session total ${client.spent_usd:.6f}")
    except NoEligibleModelError as err:
        print(f"\n{err}")


if __name__ == "__main__":
    main()
