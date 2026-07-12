"""Append-only JSONL event stream (.onramp/events.jsonl by default).

Every onboarding, probe result, routing decision, and failover is emitted
here so dashboards (e.g. the self-learning-llm-training one) can tail a
single file instead of instrumenting each subsystem."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .paths import events_path


def emit(kind: str, **fields) -> dict:
    event = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        **fields,
    }
    path = events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(event) + "\n")
    return event


def tail(n: int = 20) -> list[dict]:
    path = events_path()
    if not path.exists():
        return []
    lines = path.read_text().splitlines()[-n:]
    return [json.loads(line) for line in lines]
