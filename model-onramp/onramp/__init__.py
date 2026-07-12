"""model-onramp: pluggable model-onboarding infrastructure.

Models are plugins; infrastructure is permanent. Add a one-file adapter,
run the probes, and everything built on the registry sees the new model.
"""

from .adapter import AdapterBase, ChatResult, ModelAdapter, Pricing
from .budget import BudgetExceededError, CostTracker
from .capabilities import CapabilityManifest, detect_drift
from .client import AllCandidatesFailedError, OnrampClient
from .registry import Registry, get_registry, register, unregister
from .routing import NoEligibleModelError, RoleProfile, Router
from .stats import StatsStore, get_stats

__all__ = [
    "AdapterBase",
    "AllCandidatesFailedError",
    "BudgetExceededError",
    "CapabilityManifest",
    "ChatResult",
    "CostTracker",
    "ModelAdapter",
    "NoEligibleModelError",
    "OnrampClient",
    "Pricing",
    "Registry",
    "RoleProfile",
    "Router",
    "StatsStore",
    "detect_drift",
    "get_registry",
    "get_stats",
    "register",
    "unregister",
]
