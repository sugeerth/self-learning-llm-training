"""Capability-based routing (Phase 2).

Roles declare capability *requirements*, never model names. The Router
resolves a role to a ranked candidate list against live manifests, so a
model onboarded five minutes ago is immediately eligible for every role
it qualifies for."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .capabilities import CapabilityManifest
from .events import emit
from .registry import Registry, get_registry


@dataclass
class RoleProfile:
    name: str
    needs: dict = field(default_factory=dict)
    prefer: str = "cost"  # "cost" (cheapest first) or "speed" (fastest first)


DEFAULT_ROLES = {
    # Judges must produce machine-parseable verdicts reliably.
    "judge": RoleProfile("judge",
                         needs={"json_reliability": 0.9,
                                "instruction_score": 0.8}),
    # The MetaJudge audits long transcripts.
    "meta_judge": RoleProfile("meta_judge",
                              needs={"json_reliability": 0.9,
                                     "usable_context_tokens": 16_000}),
    # Trainers generate lots of tokens: cheap and steerable.
    "trainer": RoleProfile("trainer",
                           needs={"instruction_score": 0.6}),
    # Interactive drafting favors speed over cost.
    "drafter": RoleProfile("drafter",
                           needs={"instruction_score": 0.6},
                           prefer="speed"),
    # Agents that drive tools.
    "tool_agent": RoleProfile("tool_agent",
                              needs={"tool_use_reliability": 0.9}),
}


def load_roles(path: str | Path | None = None) -> dict[str, RoleProfile]:
    """Defaults, optionally overridden/extended by a roles.json file:
    {"judge": {"needs": {"json_reliability": 0.99}, "prefer": "cost"}}"""
    roles = dict(DEFAULT_ROLES)
    path = Path(path) if path else Path.cwd() / "roles.json"
    if path.exists():
        for name, spec in json.loads(path.read_text()).items():
            roles[name] = RoleProfile(name, needs=spec.get("needs", {}),
                                      prefer=spec.get("prefer", "cost"))
    return roles


class NoEligibleModelError(RuntimeError):
    def __init__(self, role: RoleProfile):
        super().__init__(
            f"no probed model satisfies role '{role.name}' (needs "
            f"{role.needs}) — onboard one: python -m onramp probe <model-id>")


class Router:
    def __init__(self, registry: Registry | None = None,
                 roles: dict[str, RoleProfile] | None = None):
        self.registry = registry or get_registry()
        self.roles = roles or load_roles()

    def _rank_key(self, manifest: CapabilityManifest, prefer: str):
        if prefer == "speed":
            return -(manifest.tokens_per_second or 0.0)
        return manifest.output_per_mtok or float("inf")

    def candidates(self, role_name: str) -> list[str]:
        """All eligible models for a role, best-ranked first. This IS the
        fallback chain: the client walks it in order on failure."""
        role = self.roles[role_name]
        eligible = []
        for model_id in self.registry.model_ids():
            manifest = self.registry.manifest(model_id)
            if manifest and manifest.satisfies(**role.needs):
                eligible.append((self._rank_key(manifest, role.prefer), model_id))
        eligible.sort()
        ranked = [model_id for _, model_id in eligible]
        emit("route", role=role_name, candidates=ranked)
        return ranked

    def resolve(self, role_name: str) -> str:
        ranked = self.candidates(role_name)
        if not ranked:
            raise NoEligibleModelError(self.roles[role_name])
        return ranked[0]
