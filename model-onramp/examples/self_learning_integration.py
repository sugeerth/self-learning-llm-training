"""How self-learning-llm-training plugs into the on-ramp (Phase 3 preview).

Instead of hard-coding a model id per agent, each agent role declares the
capabilities it needs and resolves a model at runtime. When a new model is
onboarded (one adapter file + a probe run), it becomes eligible for every
role it qualifies for — with zero changes in the self-learning repo.
"""

from onramp import get_registry

# Role profiles: capability requirements, not model names.
ROLE_PROFILES = {
    # Judges must produce machine-parseable verdicts reliably.
    "judge": dict(json_reliability=0.95),
    # The MetaJudge audits long transcripts, so favor context (Phase 1 probe).
    "meta_judge": dict(json_reliability=0.95),
    # Trainers generate lots of tokens; find() already ranks cheapest-first.
    "trainer": dict(json_reliability=0.8),
}


def resolve(role: str) -> str:
    registry = get_registry()
    candidates = registry.find(**ROLE_PROFILES[role])
    if not candidates:
        raise RuntimeError(
            f"no probed model satisfies role '{role}' — onboard one with "
            f"'python -m onramp probe <model-id>'"
        )
    return candidates[0]


if __name__ == "__main__":
    for role in ROLE_PROFILES:
        try:
            print(f"{role:<12} -> {resolve(role)}")
        except RuntimeError as err:
            print(f"{role:<12} -> {err}")
