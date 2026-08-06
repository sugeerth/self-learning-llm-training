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


def consensus_plan(memory: TrajectoryMemory, domain: str, goal: str,
                   k: int = 5) -> list[str] | None:
    """The tool sequence the retrieved high-quality experiences agree on.
    Adjacent duplicates (fail->retry artifacts in the data) are collapsed;
    off-domain tools are stripped. None when no same-domain exemplar exists
    — the caller falls back to the domain default."""
    hits = memory.retrieve(f"{domain} {goal}", k=k)
    seqs = [tuple(t.tools) for t, _ in hits if t.domain == domain and t.success]
    if not seqs:
        return None
    best_seq = Counter(seqs).most_common(1)[0][0]
    relevant = set(DOMAIN_TOOLS[domain])
    plan: list[str] = []
    for tool in best_seq:
        if tool in relevant and (not plan or plan[-1] != tool):
            plan.append(tool)
    return plan


class TrajectoryLearnedAgent:
    name = "trajectory-learned"

    def __init__(self, memory: TrajectoryMemory, k: int = 5):
        self.memory = memory
        self.k = k

    def act(self, domain: str, goal: str) -> Rollout:
        plan = consensus_plan(self.memory, domain, goal, k=self.k)
        if plan is None:
            plan = list(FAMILIES[(domain, _DEFAULT_FAMILY[domain])])
        return _rollout(plan)


# ────────────────── closed-loop policies (ToolEnv, v2) ──────────────────
# In the stochastic environment an agent is a POLICY: it sees each step's
# outcome and picks the next action. The four below form the ablation grid —
# plans and recovery can be switched independently, so the evaluation can
# attribute the lift to each learned skill.

class PlanPolicy:
    """Walk a plan; on a failed step, retry it if the mined RecoveryRule says
    retrying flaky tools works (and within its retry budget), else move on —
    exactly what a policy that never learned recovery would do."""

    name = "plan"

    def __init__(self, recovery=None):
        self.recovery = recovery      # mining.RecoveryRule | None

    def _plan(self, domain: str, goal: str) -> list[str]:
        return list(FAMILIES[(domain, _DEFAULT_FAMILY[domain])])

    def reset(self, domain: str, goal: str) -> None:
        self.plan = self._plan(domain, goal)
        self.i = 0
        self.retries_here = 0

    def next_tool(self, history: list[tuple[str, bool]]) -> str | None:
        if (history and not history[-1][1] and self.recovery
                and self.recovery.retry
                and self.retries_here < self.recovery.max_retries):
            self.retries_here += 1
            return history[-1][0]                 # retry the failed tool
        self.retries_here = 0
        if self.i >= len(self.plan):
            return None
        tool = self.plan[self.i]
        self.i += 1
        return tool


class BaselinePolicy(PlanPolicy):
    """Memoryless AND recovery-blind: default plan, never retries."""
    name = "baseline"

    def __init__(self):
        super().__init__(recovery=None)


class LearnedPolicy(PlanPolicy):
    """The full trajectory-learned agent: plans retrieved from experience,
    recovery rule mined from experience."""
    name = "trajectory-learned"

    def __init__(self, memory: TrajectoryMemory, recovery=None, k: int = 5):
        super().__init__(recovery=recovery)
        self.memory = memory
        self.k = k

    def _plan(self, domain: str, goal: str) -> list[str]:
        plan = consensus_plan(self.memory, domain, goal, k=self.k)
        return plan if plan is not None else super()._plan(domain, goal)


def ablation_policies(memory: TrajectoryMemory, recovery, k: int = 5) -> dict:
    """name -> policy, the 2x2 grid of (plans, recovery)."""
    return {
        "baseline": BaselinePolicy(),
        "+plans": LearnedPolicy(memory, recovery=None, k=k),
        "+recovery": PlanPolicy(recovery=recovery),
        "learned (full)": LearnedPolicy(memory, recovery=recovery, k=k),
    }
