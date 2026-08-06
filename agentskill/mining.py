"""Skill mining — extract reusable behaviors from trajectories.

Nothing here is hardcoded into the agent: the recovery rule and the sub-skill
inventory are STATISTICS OF THE DATA. Change the trajectories and the learned
behavior changes with them — that's the "learning from public trajectories"
claim made concrete and testable.

  mine_recovery()  "when a tool fails transiently, does retrying it work?"
                   Counts retry events (a failed step immediately followed by
                   the same tool) across trajectories and their success rate.
                   Only if the evidence is strong does the learned policy get
                   a retry rule. Domain-agnostic on purpose — this is the
                   procedural skill that should TRANSFER across domains.

  mine_skills()    frequent tool n-grams among successful steps, per domain —
                   the reusable sub-skills (e.g. swe: edit->run_tests) that a
                   composition-based planner can assemble.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .trajectory import Trajectory


@dataclass
class RecoveryRule:
    retry: bool                # should the policy retry a failed tool?
    max_retries: int
    p_success: float           # observed P(retry succeeds | retry attempted)
    support: int               # number of retry events observed

    def as_dict(self) -> dict:
        return {"retry": self.retry, "max_retries": self.max_retries,
                "p_success": round(self.p_success, 3), "support": self.support}


def mine_recovery(trajs: list[Trajectory], min_support: int = 5,
                  min_p: float = 0.7) -> RecoveryRule:
    """Scan for failure->same-tool-retry events and measure their success."""
    events = successes = 0
    for t in trajs:
        steps = t.steps
        for i in range(len(steps) - 1):
            if not steps[i].ok and steps[i + 1].tool == steps[i].tool:
                events += 1
                successes += int(steps[i + 1].ok)
    p = successes / events if events else 0.0
    return RecoveryRule(retry=(events >= min_support and p >= min_p),
                        max_retries=1, p_success=p, support=events)


def mine_skills(trajs: list[Trajectory], n: int = 2, top: int = 6
                ) -> dict[str, list[tuple[tuple[str, ...], int]]]:
    """Per-domain frequent n-grams over the SUCCESSFUL steps of successful
    trajectories — the sub-skill inventory."""
    grams: dict[str, Counter] = defaultdict(Counter)
    for t in trajs:
        if not t.success:
            continue
        ok_tools = [s.tool for s in t.steps if s.ok]
        for i in range(len(ok_tools) - n + 1):
            grams[t.domain][tuple(ok_tools[i:i + n])] += 1
    return {d: c.most_common(top) for d, c in grams.items()}
