"""model-onramp: pluggable model-onboarding infrastructure.

Models are plugins; infrastructure is permanent. Add a one-file adapter,
run the probes, and everything built on the registry sees the new model.
"""

from .adapter import ModelAdapter, Pricing
from .capabilities import CapabilityManifest
from .registry import Registry, register, get_registry

__all__ = [
    "ModelAdapter",
    "Pricing",
    "CapabilityManifest",
    "Registry",
    "register",
    "get_registry",
]
