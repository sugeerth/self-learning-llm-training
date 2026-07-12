"""Hard spend caps. Probe runs and client sessions charge every call to a
CostTracker; exceeding the cap raises rather than silently burning money."""

from __future__ import annotations

from .adapter import ChatResult


class BudgetExceededError(RuntimeError):
    def __init__(self, spent_usd: float, cap_usd: float):
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd
        super().__init__(
            f"budget exceeded: ${spent_usd:.4f} spent, cap ${cap_usd:.4f}")


class CostTracker:
    def __init__(self, cap_usd: float | None = None):
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self.calls = 0

    def charge(self, result: ChatResult) -> None:
        self.spent_usd += result.cost_usd
        self.calls += 1
        if self.cap_usd is not None and self.spent_usd > self.cap_usd:
            raise BudgetExceededError(self.spent_usd, self.cap_usd)

    def would_exceed(self, estimated_usd: float) -> bool:
        return (self.cap_usd is not None
                and self.spent_usd + estimated_usd > self.cap_usd)
