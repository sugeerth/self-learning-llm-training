"""Capability manifests: measured, versioned descriptions of what a model
can actually do. Downstream infrastructure consumes these instead of model
names, which is what lets a model released tomorrow work with code written
today."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Bump when the probe suite changes; cached manifests from older suite
# versions are considered stale and re-probed.
PROBE_SUITE_VERSION = 1

MANIFEST_DIR = Path(__file__).resolve().parent.parent / "manifests"


@dataclass
class CapabilityManifest:
    model_id: str
    probe_suite_version: int = PROBE_SUITE_VERSION

    # Measured by probes (None = not yet probed)
    usable_context_tokens: int | None = None
    json_reliability: float | None = None      # 0..1 parse-success rate
    tool_use: bool | None = None
    instruction_score: float | None = None     # 0..1 rubric score
    tokens_per_second: float | None = None

    # Declared by the adapter
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None

    notes: dict = field(default_factory=dict)

    def satisfies(self, **needs) -> bool:
        """Check requirements like json_reliability=0.95 (min for floats/ints,
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
        return MANIFEST_DIR / f"{self.model_id}.json"

    def save(self) -> Path:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return self.path

    @classmethod
    def load(cls, model_id: str) -> "CapabilityManifest | None":
        path = MANIFEST_DIR / f"{model_id}.json"
        if not path.exists():
            return None
        manifest = cls(**json.loads(path.read_text()))
        if manifest.probe_suite_version != PROBE_SUITE_VERSION:
            return None  # stale: probe suite has changed since it was measured
        return manifest
