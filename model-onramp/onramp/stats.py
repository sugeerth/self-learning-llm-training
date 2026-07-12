"""Live outcome statistics: what production traffic teaches us about each
model, beyond what probes measured at onboarding.

Every OnrampClient call records success/failure, cost, and latency here
(per model, per role). Consumers can add quality scores via
client.feedback(). These stats drive:

  - bandit exploration   (candidates earn live traffic)
  - the circuit breaker  (failing models are skipped until they cool down)
  - autopilot            (auto-promote / auto-demote on live evidence)
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .paths import onramp_home


def _blank_role() -> dict:
    return {"calls": 0, "successes": 0, "failures": 0,
            "cost_usd": 0.0, "latency_s": 0.0,
            "score_sum": 0.0, "score_n": 0,
            "consecutive_failures": 0, "last_failure_ts": 0.0}


AGGREGATE = "*"  # pseudo-role holding the per-model rollup


class StatsStore:
    def __init__(self, path: str | Path | None = None):
        self._lock = threading.Lock()
        self._path = Path(path) if path else onramp_home() / "live_stats.json"
        self._data: dict[str, dict[str, dict]] = {}
        if self._path.exists():
            self._data = json.loads(self._path.read_text())

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2) + "\n")

    def _bucket(self, model_id: str, role: str) -> dict:
        return self._data.setdefault(model_id, {}).setdefault(role, _blank_role())

    # -- recording -------------------------------------------------------

    def record_call(self, model_id: str, role: str | None, success: bool,
                    cost_usd: float = 0.0, latency_s: float = 0.0,
                    now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            for key in {role or AGGREGATE, AGGREGATE}:
                bucket = self._bucket(model_id, key)
                bucket["calls"] += 1
                bucket["cost_usd"] += cost_usd
                bucket["latency_s"] += latency_s
                if success:
                    bucket["successes"] += 1
                    bucket["consecutive_failures"] = 0
                else:
                    bucket["failures"] += 1
                    bucket["consecutive_failures"] += 1
                    bucket["last_failure_ts"] = now
            self._save()

    def record_score(self, model_id: str, role: str | None,
                     score: float) -> None:
        """Quality feedback in [0, 1] from the consumer (e.g. a judge's
        verdict on output produced by this model)."""
        score = max(0.0, min(1.0, score))
        with self._lock:
            for key in {role or AGGREGATE, AGGREGATE}:
                bucket = self._bucket(model_id, key)
                bucket["score_sum"] += score
                bucket["score_n"] += 1
            self._save()

    # -- reading ---------------------------------------------------------

    def get(self, model_id: str, role: str | None = None) -> dict:
        return dict(self._data.get(model_id, {}).get(role or AGGREGATE,
                                                     _blank_role()))

    def calls(self, model_id: str, role: str | None = None) -> int:
        return self.get(model_id, role)["calls"]

    def success_rate(self, model_id: str, role: str | None = None,
                     prior: tuple[int, int] = (1, 1)) -> float:
        """Laplace-smoothed success rate — 0.5 with no data under the
        default (1,1) prior, converging to the observed rate."""
        bucket = self.get(model_id, role)
        alpha, beta = prior
        return ((bucket["successes"] + alpha)
                / (bucket["calls"] + alpha + beta))

    def mean_score(self, model_id: str, role: str | None = None) -> float | None:
        bucket = self.get(model_id, role)
        if not bucket["score_n"]:
            return None
        return bucket["score_sum"] / bucket["score_n"]

    def breaker_open(self, model_id: str, *, failure_threshold: int = 3,
                     cooldown_s: float = 60.0,
                     now: float | None = None) -> bool:
        """Circuit breaker: open (skip this model) after `failure_threshold`
        consecutive failures, until `cooldown_s` has passed since the last
        one — then half-open: the next attempt is allowed through."""
        bucket = self.get(model_id)
        if bucket["consecutive_failures"] < failure_threshold:
            return False
        now = time.time() if now is None else now
        return (now - bucket["last_failure_ts"]) < cooldown_s


_store: StatsStore | None = None


def get_stats() -> StatsStore:
    global _store
    if _store is None or _store._path != (onramp_home() / "live_stats.json"):
        _store = StatsStore()  # re-open when ONRAMP_HOME changes (tests)
    return _store
