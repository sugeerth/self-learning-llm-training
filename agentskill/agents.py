"""Two agents to compare.

BaselineAgent — memoryless. It knows each domain's tools and a sensible default
plan, but it can't tell task families apart, so it applies the same plan to
every task in a domain. This stands in for a capable but un-adapted agent.

TrajectoryLearnedAgent — retrieves the most similar high-quality past
trajectories from memory and imitates the tool sequence they agree on
(retrieval-augmented imitation / nearest-neighbour policy). This is the
offline-demonstrable form of "learning from public trajectories": planning and
tool selection are transferred from experience, not hard-coded. The LoRA
fine-tune (finetune.py, GPU/Colab) distills the same curated experiences into
the weights; this agent shows the lift without needing a GPU.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .memory import TrajectoryMemory
from .trajectory import DOMAIN_TOOLS, _DEFAULT_FAMILY, FAMILIES


@dataclass
class Rollout:
    tools: list[str]
    n_steps: int
    cost_usd: float
    runtime_s: float


STEP_COST = 0.002       # $/step
STEP_TIME = 1.5         # s/step


def _rollout(tools: list[str]) -> Rollout:
    n = len(tools)
    return Rollout(tools=tools, n_steps=n, cost_usd=round(STEP_COST * n, 4),
                   runtime_s=round(STEP_TIME * n, 1))


class BaselineAgent:
    name = "baseline"

    def act(self, domain: str, goal: str) -> Rollout:
        # always the domain's default-family plan — no adaptation to the task
        plan = FAMILIES[(domain, _DEFAULT_FAMILY[domain])]
        return _rollout(list(plan))


class TrajectoryLearnedAgent:
    name = "trajectory-learned"

    def __init__(self, memory: TrajectoryMemory, k: int = 5):
        self.memory = memory
        self.k = k

    def act(self, domain: str, goal: str) -> Rollout:
        hits = self.memory.retrieve(f"{domain} {goal}", k=self.k)
        # keep same-domain, successful exemplars
        seqs = [tuple(t.tools) for t, _ in hits
                if t.domain == domain and t.success]
        if not seqs:
            plan = FAMILIES[(domain, _DEFAULT_FAMILY[domain])]
            return _rollout(list(plan))
        # imitate the tool sequence the retrieved experiences agree on, and
        # strip the known dead-end first-steps recovered trajectories carry
        best_seq = Counter(seqs).most_common(1)[0][0]
        relevant = set(DOMAIN_TOOLS[domain])
        plan = [t for t in best_seq if t in relevant]
        return _rollout(plan)
