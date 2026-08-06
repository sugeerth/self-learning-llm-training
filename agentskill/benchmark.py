"""Benchmark tasks + scoring, standing in for GAIA / MLE-bench / SWE-bench.

Each Task has a hidden canonical tool sequence (its family's optimal plan). An
agent's rollout SOLVES the task when the canonical sequence appears in order in
its actions (subsequence match) — captures "did it take the necessary steps",
tolerant of an extra exploratory step. The real benchmarks plug in behind the
same `Task`/`grade` interface (see agentskill/README.md → Real benchmarks).

Offline: `mock_suite()` samples held-out tasks from the same families as the
trajectory memory — the transfer setting the project asks about (new tasks,
same skills).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .trajectory import FAMILIES, _GOALS, canonical_for

# which real benchmark each domain stands in for
BENCHMARKS = {"gaia": "GAIA", "mle": "MLE-bench", "swe": "SWE-bench Verified"}


@dataclass
class Task:
    task_id: str
    domain: str
    goal: str

    @property
    def canonical(self) -> list[str]:
        return canonical_for(self.domain, self.goal)


def _is_subsequence(needle: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(tok in it for tok in needle)


def grade(task: Task, tools: list[str]) -> dict:
    """Return solved + the metrics the project tracks for one task."""
    canon = task.canonical
    solved = _is_subsequence(canon, tools)
    n = max(len(tools), 1)
    # tool efficiency: share of the agent's actions that are canonical steps
    tool_eff = sum(t in canon for t in tools) / n
    return {"solved": solved, "steps": len(tools),
            "tool_efficiency": round(tool_eff, 3),
            "cost_usd": round(0.002 * len(tools), 4),
            "runtime_s": round(1.5 * len(tools), 1)}


def mock_suite(n_per_family: int = 10, seed: int = 100) -> list[Task]:
    """Held-out evaluation tasks — same families as memory, new instances
    (task ids offset so they never coincide with the training trajectories)."""
    rng = random.Random(seed)
    tasks = []
    for (domain, family) in FAMILIES:
        for i in range(n_per_family):
            goal = f"{rng.choice(_GOALS[(domain, family)])} (eval {i})"
            tasks.append(Task(task_id=f"eval-{domain}-{family}-{i}",
                              domain=domain, goal=goal))
    return tasks
