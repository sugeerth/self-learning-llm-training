from onramp import CapabilityManifest, unregister
from onramp.discovery import discover


def test_discover_registers_only_new_models(registry):
    fake_api = ["claude-opus-4-8",          # already registered -> skipped
                "claude-weird-experiment"]  # new, no pricing prefix matches
    new_ids = discover(lister=lambda: fake_api)
    try:
        assert new_ids == ["claude-weird-experiment"]
        assert "claude-weird-experiment" in registry

        # unknown pricing -> $0 placeholder + flag that blocks promotion
        pricing = registry.get("claude-weird-experiment").pricing
        assert (pricing.input_per_mtok, pricing.output_per_mtok) == (0.0, 0.0)
        manifest = CapabilityManifest.load("claude-weird-experiment")
        assert manifest.notes.get("pricing_unknown") is True

        # idempotent: second run discovers nothing
        assert discover(lister=lambda: fake_api) == []
    finally:
        unregister("claude-weird-experiment")


def test_discover_prefix_pricing(registry):
    # a hypothetical future Haiku matches the "claude-haiku-4" pricing prefix
    new_ids = discover(lister=lambda: ["claude-haiku-4-9"])
    try:
        assert new_ids == ["claude-haiku-4-9"]
        pricing = registry.get("claude-haiku-4-9").pricing
        assert (pricing.input_per_mtok, pricing.output_per_mtok) == (1.00, 5.00)
        # priced models carry no pricing_unknown flag
        manifest = CapabilityManifest.load("claude-haiku-4-9")
        assert manifest is None or not manifest.notes.get("pricing_unknown")
    finally:
        unregister("claude-haiku-4-9")
