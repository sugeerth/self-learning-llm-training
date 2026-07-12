"""OnrampClient: the one call site downstream infrastructure needs.

client.chat(messages, role="judge") resolves the role to a fallback chain,
walks it on provider failure, charges every call to a session cost cap,
and emits events — so consumers get routing, failover, and accounting
without knowing any model names."""

from __future__ import annotations

from .adapter import ChatResult
from .budget import CostTracker
from .events import emit
from .routing import NoEligibleModelError, Router


class AllCandidatesFailedError(RuntimeError):
    def __init__(self, role: str, errors: dict[str, str]):
        self.errors = errors
        detail = "; ".join(f"{m}: {e}" for m, e in errors.items())
        super().__init__(f"every candidate for role '{role}' failed — {detail}")


class OnrampClient:
    def __init__(self, router: Router | None = None,
                 cost_cap_usd: float | None = None):
        self.router = router or Router()
        self.tracker = CostTracker(cap_usd=cost_cap_usd)

    @property
    def spent_usd(self) -> float:
        return self.tracker.spent_usd

    def chat(self, messages: list[dict], *, role: str | None = None,
             model_id: str | None = None, max_tokens: int = 1024,
             temperature: float = 0.0) -> ChatResult:
        """Route by role (with failover) or pin an explicit model_id."""
        if (role is None) == (model_id is None):
            raise ValueError("pass exactly one of role= or model_id=")

        candidates = [model_id] if model_id else self.router.candidates(role)
        if not candidates:
            raise NoEligibleModelError(self.router.roles[role])

        errors: dict[str, str] = {}
        for candidate in candidates:
            adapter = self.router.registry.get(candidate)
            try:
                result = adapter.chat(messages, max_tokens=max_tokens,
                                      temperature=temperature)
            except Exception as err:  # provider outage, rate limit, etc.
                errors[candidate] = f"{type(err).__name__}: {err}"
                emit("failover", role=role, failed=candidate, error=str(err))
                continue
            self.tracker.charge(result)
            emit("chat", role=role, model_id=candidate,
                 cost_usd=round(result.cost_usd, 6),
                 latency_s=round(result.latency_s, 3))
            return result
        raise AllCandidatesFailedError(role or model_id, errors)

    def generate(self, prompt: str, **kwargs) -> ChatResult:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)
