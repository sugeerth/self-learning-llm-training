import pytest

from onramp import Pricing, register, unregister
from onramp.testing import make_mock


def test_discovery_finds_claude_adapters(registry):
    ids = registry.model_ids()
    assert "claude-opus-4-8" in ids
    assert "claude-sonnet-5" in ids
    assert "claude-haiku-4-5" in ids


def test_register_and_unregister_mock(registry):
    register(make_mock("mock-a"))
    assert "mock-a" in registry
    adapter = registry.get("mock-a")
    assert adapter.pricing == Pricing(1.0, 2.0)
    unregister("mock-a")
    assert "mock-a" not in registry


def test_get_unknown_model_raises(registry):
    with pytest.raises(KeyError, match="unknown model"):
        registry.get("no-such-model")


def test_find_excludes_unprobed_models(registry):
    register(make_mock("mock-unprobed"))
    try:
        assert "mock-unprobed" not in registry.find(json_reliability=0.5)
    finally:
        unregister("mock-unprobed")
