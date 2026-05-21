"""Braintrust integration: send traces + fetch them back for the dashboard.

Two paths:
  1. log_event / @traced  → push to Braintrust (and to local state for dashboard)
  2. fetch_recent_logs    → pull traces back via Braintrust REST API

Designed to degrade gracefully: if BRAINTRUST_API_KEY is unset or the SDK is
missing, everything still writes to local state and the run keeps going.
"""
from __future__ import annotations

import functools
import json
import os
import threading
import time
from typing import Any, Callable, Optional

import requests

PROJECT = os.getenv("BRAINTRUST_PROJECT", "llm-training-self-learning")
STATE_PATH = "/tmp/llm_training_state.jsonl"
_state_lock = threading.Lock()

# ── braintrust SDK is optional ─────────────────────────────────────────
try:
    import braintrust  # type: ignore
    _bt_logger = braintrust.init_logger(project=PROJECT) if os.getenv("BRAINTRUST_API_KEY") else None
except Exception:
    braintrust = None  # type: ignore
    _bt_logger = None


def _local_write(record: dict) -> None:
    record["ts"] = record.get("ts") or time.time()
    line = json.dumps(record, default=str) + "\n"
    with _state_lock:
        with open(STATE_PATH, "a") as f:
            f.write(line)


def log_event(**fields: Any) -> None:
    """Log a structured event to Braintrust (if configured) AND local state."""
    _local_write({"kind": "event", **fields})
    if _bt_logger is not None:
        try:
            _bt_logger.log(input=fields.get("input", fields), output=fields.get("output"),
                           metadata={k: v for k, v in fields.items() if k not in ("input", "output")})
        except Exception as e:  # never let logging break the run
            _local_write({"kind": "log_error", "error": str(e)})


def traced(span_name: str) -> Callable:
    """Decorator: wrap a function in a Braintrust span (or a no-op if disabled)."""
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            t0 = time.time()
            _local_write({"kind": "span_start", "span": span_name})
            try:
                if braintrust is not None and _bt_logger is not None:
                    with braintrust.start_span(name=span_name) as span:  # type: ignore
                        result = fn(*args, **kwargs)
                        try:
                            span.log(input=str(args)[:1000], output=str(result)[:2000])
                        except Exception:
                            pass
                        return result
                return fn(*args, **kwargs)
            finally:
                _local_write({
                    "kind": "span_end", "span": span_name,
                    "duration_ms": int((time.time() - t0) * 1000),
                })
        return wrapped
    return deco


# ── pulling traces back ────────────────────────────────────────────────

def fetch_recent_logs(limit: int = 50) -> list[dict]:
    """Best-effort fetch from Braintrust REST API. Falls back to local state file."""
    key = os.getenv("BRAINTRUST_API_KEY")
    if key:
        try:
            r = requests.get(
                f"https://api.braintrust.dev/v1/project_logs/{PROJECT}/fetch",
                headers={"Authorization": f"Bearer {key}"},
                params={"limit": limit},
                timeout=8,
            )
            if r.ok:
                return r.json().get("events", [])
        except Exception:
            pass
    # fallback: read local state file (last `limit` lines)
    try:
        with open(STATE_PATH) as f:
            lines = f.readlines()[-limit:]
        return [json.loads(l) for l in lines if l.strip()]
    except FileNotFoundError:
        return []


def reset_state() -> None:
    """Clear the local state file (called at the start of a fresh run)."""
    try:
        os.remove(STATE_PATH)
    except FileNotFoundError:
        pass


def write_snapshot(snapshot: dict) -> None:
    """Atomic write of the current run snapshot (consumed by dashboard /api/state)."""
    tmp = "/tmp/llm_training_snapshot.json.tmp"
    final = "/tmp/llm_training_snapshot.json"
    with open(tmp, "w") as f:
        json.dump(snapshot, f, default=str)
    os.replace(tmp, final)


def read_snapshot() -> dict:
    try:
        with open("/tmp/llm_training_snapshot.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def read_human_queue() -> list[dict]:
    try:
        with open("/tmp/llm_training_human_queue.jsonl") as f:
            return [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return []
