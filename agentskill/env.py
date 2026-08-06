"""ToolEnv — a closed-loop, stochastic tool environment.

The v1 evaluation graded open-loop plans in a noise-free world, so "debugging
and recovery" was never actually exercised at eval time. Here, every tool call
can fail TRANSIENTLY (seeded per task, deterministic per seed): the same tool
retried immediately after its failure succeeds — the classic flaky-tool /
flaky-test model. An agent is now a POLICY that observes each outcome and
chooses the next action, so recovery is a real, measured behavior:

  - a policy that ignores failures loses that canonical step and fails
  - a policy that learned "on transient failure, retry" recovers at the cost
    of an extra step (which the cost/efficiency metrics then account for)

A task is solved when its canonical tool sequence appears, in order, among the
SUCCESSFUL steps of the episode.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Protocol

from .benchmark import Task, _is_subsequence


class Policy(Protocol):
    def reset(self, domain: str, goal: str) -> None: ...
    def next_tool(self, history: list[tuple[str, bool]]) -> str | None: ...


@dataclass
class Episode:
    task: Task
    steps: list[tuple[str, bool]] = field(default_factory=list)   # (tool, ok)

    @property
    def ok_tools(self) -> list[str]:
        return [t for t, ok in self.steps if ok]

    @property
    def solved(self) -> bool:
        return _is_subsequence(self.task.canonical, self.ok_tools)

    def metrics(self) -> dict:
        n = max(len(self.steps), 1)
        canon = set(self.task.canonical)
        return {
            "solved": self.solved,
            "steps": len(self.steps),
            "tool_efficiency": round(
                sum(t in canon for t, _ in self.steps) / n, 3),
            "cost_usd": round(0.002 * len(self.steps), 4),
            "runtime_s": round(1.5 * len(self.steps), 1),
        }


class ToolEnv:
    def __init__(self, task: Task, fail_rate: float = 0.15, seed: int = 0,
                 max_steps: int | None = None):
        self.task = task
        self.fail_rate = fail_rate
        self.max_steps = max_steps or (3 * len(task.canonical) + 2)
        digest = hashlib.sha256(f"{task.task_id}|{seed}".encode()).hexdigest()
        self.rng = random.Random(int(digest[:16], 16))

    def run(self, policy: Policy) -> Episode:
        policy.reset(self.task.domain, self.task.goal)
        ep = Episode(task=self.task)
        while len(ep.steps) < self.max_steps:
            tool = policy.next_tool(ep.steps)
            if tool is None:
                break
            if ep.steps and ep.steps[-1] == (tool, False):
                ok = True          # transient: immediate retry succeeds
            else:
                ok = self.rng.random() >= self.fail_rate
            ep.steps.append((tool, ok))
        return ep
