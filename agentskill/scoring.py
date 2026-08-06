"""Trajectory quality scoring — which experiences are worth learning from.

Offline, deterministic, no model required (a learned ranker is the GPU upgrade
path; this heuristic is the always-available baseline and the label source the
ranker distills from). The score rewards exactly the skills the project targets:

  - success            did it actually solve the task?
  - efficiency         optimal path length / actual length (planning)
  - tool relevance     share of actions that are on-domain (tool selection)
  - recovery           bonus when a wrong step was followed by success
                       (learning to recover from mistakes)

Failed trajectories are kept but heavily down-weighted — they still teach what
not to do, which the spec explicitly asks for.
"""

from __future__ import annotations

from dataclasses import dataclass

from .trajectory import DOMAIN_TOOLS, Trajectory, canonical_for


@dataclass
class QualityScore:
    total: float               # 0..1
    success: float
    efficiency: float
    tool_relevance: float
    recovery: float

    def as_dict(self) -> dict:
        return {"total": round(self.total, 3), "success": self.success,
                "efficiency": round(self.efficiency, 3),
                "tool_relevance": round(self.tool_relevance, 3),
                "recovery": round(self.recovery, 3)}


def score_trajectory(t: Trajectory) -> QualityScore:
    tools = t.tools
    n = max(len(tools), 1)
    optimal = len(canonical_for(t.domain, t.goal))

    success = 1.0 if t.success else 0.0
    efficiency = min(1.0, optimal / n) if t.success else 0.0
    relevant = set(DOMAIN_TOOLS.get(t.domain, []))
    tool_relevance = sum(tool in relevant for tool in tools) / n
    recovery = 1.0 if t.recovered else 0.0

    # weighted blend; failures retain a small floor so they're still learnable
    total = (0.55 * success + 0.20 * efficiency + 0.15 * tool_relevance
             + 0.10 * recovery)
    if not t.success:
        total = 0.15 * tool_relevance      # keep, but well below any success
    return QualityScore(total=total, success=success, efficiency=efficiency,
                        tool_relevance=tool_relevance, recovery=recovery)


def rank(trajs: list[Trajectory]) -> list[tuple[Trajectory, QualityScore]]:
    scored = [(t, score_trajectory(t)) for t in trajs]
    scored.sort(key=lambda ts: -ts[1].total)
    return scored


def curate(trajs: list[Trajectory], min_quality: float = 0.6
           ) -> list[tuple[Trajectory, QualityScore]]:
    """High-quality subset for fine-tuning / retrieval memory."""
    return [(t, s) for t, s in rank(trajs) if s.total >= min_quality]
