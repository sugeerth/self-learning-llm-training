"""Auto-discovering model registry.

Adapters self-register with @register; the registry imports every module in
onramp/adapters/ so there is no central list to edit when a model is added.
"""

from __future__ import annotations

import importlib
import pkgutil

from .adapter import ModelAdapter
from .capabilities import CapabilityManifest

_ADAPTERS: dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator used inside adapter files: @register."""
    _ADAPTERS[cls.model_id] = cls
    return cls


def unregister(model_id: str) -> None:
    """Mainly for tests and hot-swapping; removing an adapter file is the
    normal way to retire a model."""
    _ADAPTERS.pop(model_id, None)


class Registry:
    def __init__(self) -> None:
        self._discover()
        self._instances: dict[str, ModelAdapter] = {}

    def _discover(self) -> None:
        from . import adapters as pkg

        for mod in pkgutil.iter_modules(pkg.__path__):
            if not mod.name.startswith("_"):
                importlib.import_module(f"{pkg.__name__}.{mod.name}")

    # -- lookup ---------------------------------------------------------

    def model_ids(self) -> list[str]:
        return sorted(_ADAPTERS)

    def __contains__(self, model_id: str) -> bool:
        return model_id in _ADAPTERS

    def get(self, model_id: str) -> ModelAdapter:
        if model_id not in _ADAPTERS:
            raise KeyError(
                f"unknown model '{model_id}' — registered: {self.model_ids()}")
        if model_id not in self._instances:
            self._instances[model_id] = _ADAPTERS[model_id]()
        return self._instances[model_id]

    def manifest(self, model_id: str) -> CapabilityManifest | None:
        return CapabilityManifest.load(model_id)

    def find(self, sort_by_cost: bool = True, **needs) -> list[str]:
        """Capability-based routing: registry.find(json_reliability=0.95,
        usable_context_tokens=100_000) returns matching model ids, cheapest
        first. Models without a manifest (never probed) are excluded."""
        matches = []
        for model_id in self.model_ids():
            manifest = self.manifest(model_id)
            if manifest and manifest.satisfies(**needs):
                matches.append((manifest.output_per_mtok or 0.0, model_id))
        if sort_by_cost:
            matches.sort()
        return [model_id for _, model_id in matches]


_registry: Registry | None = None


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry
