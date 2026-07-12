"""OnrampClient: the one call site downstream infrastructure needs.

client.chat(messages, role="judge") resolves the role to a fallback chain
and serves the request with the full reliability stack:

  - circuit breaker : models with consecutive live failures are skipped
                      until a cooldown passes (then one probe attempt)
  - exploration     : a small share of traffic (explore_rate) goes to a
                      non-first candidate so newcomers earn live stats —
                      the data autopilot uses to auto-promote them
  - retries         : transient failures retry with exponential backoff
  - failover        : then the next candidate in the chain serves
  - cost cap        : every call charges a per-session budget
  - outcomes        : every success/failure lands in the live StatsStore;
                      client.feedback() adds quality scores

Consumers never see a model name unless they ask for one.
"""

from __future__ import annotations

import random
import time

from .adapter import ChatResult
from .budget import CostTracker
from .events import emit
from .routing import NoEligibleModelError, Router
from .stats import StatsStore, get_stats


class AllCandidatesFailedError(RuntimeError):
    def __init__(self, role: str, errors: dict[str, str]):
        self.errors = errors
        detail = "; ".join(f"{m}: {e}" for m, e in errors.items())
        super().__init__(f"every candidate for role '{role}' failed — {detail}")


class OnrampClient:
    def __init__(self, router: Router | None = None,
                 cost_cap_usd: float | None = None,
                 max_retries: int = 2, retry_base_delay: float = 0.5,
                 explore_rate: float = 0.0,
                 breaker_failure_threshold: int = 3,
                 breaker_cooldown_s: float = 60.0,
                 stats: StatsStore | None = None,
                 rng: random.Random | None = None):
        self.router = router or Router()
        self.tracker = CostTracker(cap_usd=cost_cap_usd)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.explore_rate = explore_rate
        self.breaker_failure_threshold = breaker_failure_threshold
        self.breaker_cooldown_s = breaker_cooldown_s
        self.stats = stats or get_stats()
        self.rng = rng or random.Random()

    @property
    def spent_usd(self) -> float:
        return self.tracker.spent_usd

    # -- candidate ordering ----------------------------------------------

    def _order_candidates(self, candidates: list[str], role: str | None) -> list[str]:
        # Circuit breaker: drop tripped models — unless that would empty the
        # chain, in which case serving degraded beats serving nothing.
        healthy = [c for c in candidates if not self.stats.breaker_open(
            c, failure_threshold=self.breaker_failure_threshold,
            cooldown_s=self.breaker_cooldown_s)]
        skipped = [c for c in candidates if c not in healthy]
        for model_id in skipped:
            emit("breaker_skip", model_id=model_id, role=role)
        ordered = healthy or candidates

        # Exploration: occasionally lead with a non-first candidate so
        # newcomers accumulate the live traffic autopilot needs.
        if (role is not None and len(ordered) > 1
                and self.rng.random() < self.explore_rate):
            pick = self.rng.choice(ordered[1:])
            ordered = [pick] + [c for c in ordered if c != pick]
            emit("explore", role=role, model_id=pick)
        return ordered

    # -- the call ----------------------------------------------------------

    def chat(self, messages: list[dict], *, role: str | None = None,
             model_id: str | None = None, max_tokens: int = 1024,
             temperature: float = 0.0) -> ChatResult:
        """Route by role (with the full reliability stack) or pin an
        explicit model_id."""
        if (role is None) == (model_id is None):
            raise ValueError("pass exactly one of role= or model_id=")

        candidates = [model_id] if model_id else self.router.candidates(role)
        if not candidates:
            raise NoEligibleModelError(self.router.roles[role])
        candidates = self._order_candidates(candidates, role)

        errors: dict[str, str] = {}
        for candidate in candidates:
            adapter = self.router.registry.get(candidate)
            result = None
            for attempt in range(self.max_retries + 1):
                try:
                    result = adapter.chat(messages, max_tokens=max_tokens,
                                          temperature=temperature)
                    break
                except Exception as err:  # provider outage, rate limit, etc.
                    errors[candidate] = f"{type(err).__name__}: {err}"
                    if attempt < self.max_retries:
                        delay = self.retry_base_delay * (2 ** attempt)
                        emit("retry", model_id=candidate, attempt=attempt + 1,
                             delay_s=delay, error=str(err))
                        time.sleep(delay)
            if result is None:  # retries exhausted -> next candidate
                self.stats.record_call(candidate, role, success=False)
                emit("failover", role=role, failed=candidate,
                     error=errors[candidate])
                continue
            self.tracker.charge(result)
            self.stats.record_call(candidate, role, success=True,
                                   cost_usd=result.cost_usd,
                                   latency_s=result.latency_s)
            emit("chat", role=role, model_id=candidate,
                 cost_usd=round(result.cost_usd, 6),
                 latency_s=round(result.latency_s, 3))
            return result
        raise AllCandidatesFailedError(role or model_id, errors)

    def generate(self, prompt: str, **kwargs) -> ChatResult:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    # -- quality feedback ---------------------------------------------------

    def feedback(self, result: ChatResult, score: float,
                 role: str | None = None) -> None:
        """Report output quality in [0, 1] — e.g. a judge's verdict on
        content this result produced. Feeds autopilot's promotion logic."""
        self.stats.record_score(result.model_id, role, score)
        emit("feedback", model_id=result.model_id, role=role,
             score=round(score, 4))
