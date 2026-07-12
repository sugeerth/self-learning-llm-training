"""Capability manifests: measured, versioned descriptions of what a model
can actually do. Downstream infrastructure consumes these instead of model
names, which is what lets a model released tomorrow work with code written
today.

Every save also appends to a per-model history, so silent model updates
(same id, new behavior) show up as manifest drift."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from .paths import history_dir, manifest_dir

# Bump when the probe suite changes; cached manifests from older suite
# versions are considered stale and re-probed.
PROBE_SUITE_VERSION = 2

# Fields compared by drift detection (all 0..1 rates or measured scalars).
DRIFT_FIELDS = ("json_reliability", "tool_use_reliability",
                "instruction_score", "usable_context_tokens",
                "tokens_per_second")


@dataclass
class CapabilityManifest:
    model_id: str
    probe_suite_version: int = PROBE_SUITE_VERSION
    probed_at: str | None = None

    # Rollout lifecycle: freshly probed models are "candidate" — routable,
    # but ranked below "stable" models until promoted (onramp promote <id>).
    status: str = "candidate"

    # Measured by probes (None = not yet probed)
    usable_context_tokens: int | None = None
    json_reliability: float | None = None       # 0..1 parse-success rate
    tool_use_reliability: float | None = None    # 0..1 well-formed-call rate
    instruction_score: float | None = None       # 0..1 deterministic rubric
    tokens_per_second: float | None = None
    probe_cost_usd: float | None = None

    # Declared by the adapter
    provider: str | None = None
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None

    notes: dict = field(default_factory=dict)

    def satisfies(self, **needs) -> bool:
        """Check requirements like json_reliability=0.95 (min for numbers,
        exact match for bools). Unprobed fields never satisfy a requirement."""
        for key, required in needs.items():
            value = getattr(self, key, None)
            if value is None:
                return False
            if isinstance(required, bool):
                if value is not required:
                    return False
            elif value < required:
                return False
        return True

    # -- persistence ---------------------------------------------------

    @property
    def path(self) -> Path:
        return manifest_dir() / f"{self.model_id}.json"

    def save(self, snapshot: bool = True) -> Path:
        """Persist the manifest; snapshot=False updates the cache without
        appending history (used for lifecycle changes like promote)."""
        if self.probed_at is None:
            self.probed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = json.dumps(asdict(self), indent=2) + "\n"
        manifest_dir().mkdir(parents=True, exist_ok=True)
        self.path.write_text(payload)
        if snapshot:
            hist = history_dir(self.model_id)
            hist.mkdir(parents=True, exist_ok=True)
            stamp = self.probed_at.replace(":", "-")
            (hist / f"{stamp}.json").write_text(payload)
        return self.path

    def set_status(self, status: str) -> None:
        if status not in ("candidate", "stable", "retired"):
            raise ValueError(f"invalid status '{status}'")
        self.status = status
        self.save(snapshot=False)

    @classmethod
    def _from_json(cls, raw: str) -> "CapabilityManifest":
        data = json.loads(raw)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, model_id: str) -> "CapabilityManifest | None":
        path = manifest_dir() / f"{model_id}.json"
        if not path.exists():
            return None
        manifest = cls._from_json(path.read_text())
        if manifest.probe_suite_version != PROBE_SUITE_VERSION:
            return None  # stale: probe suite has changed since it was measured
        return manifest

    @classmethod
    def history(cls, model_id: str) -> list["CapabilityManifest"]:
        """All snapshots for a model, oldest first."""
        hist = history_dir(model_id)
        if not hist.exists():
            return []
        return [cls._from_json(p.read_text()) for p in sorted(hist.glob("*.json"))]


def detect_drift(model_id: str, threshold: float = 0.10) -> list[str]:
    """Compare the two most recent snapshots; report fields whose relative
    change exceeds `threshold`. This is how silent model updates surface."""
    snapshots = CapabilityManifest.history(model_id)
    if len(snapshots) < 2:
        return []
    prev, curr = snapshots[-2], snapshots[-1]
    alerts = []
    for name in DRIFT_FIELDS:
        old, new = getattr(prev, name), getattr(curr, name)
        if old is None or new is None:
            continue
        baseline = abs(old) if old else 1.0
        if abs(new - old) / baseline > threshold:
            alerts.append(f"{name}: {old} -> {new}")
    return alerts
