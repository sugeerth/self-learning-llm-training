"""All on-disk state lives under one home directory (default: ./.onramp),
overridable via ONRAMP_HOME so tests and deployments can relocate it."""

from __future__ import annotations

import os
from pathlib import Path


def onramp_home() -> Path:
    return Path(os.environ.get("ONRAMP_HOME", Path.cwd() / ".onramp"))


def manifest_dir() -> Path:
    return onramp_home() / "manifests"


def history_dir(model_id: str) -> Path:
    return onramp_home() / "history" / model_id


def events_path() -> Path:
    return onramp_home() / "events.jsonl"
