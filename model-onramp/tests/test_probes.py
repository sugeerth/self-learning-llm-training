import pytest

from onramp import BudgetExceededError, CapabilityManifest, Pricing, register, unregister
from onramp.probes import run_probes
from onramp.testing import broken_responder, make_mock


@pytest.fixture
def perfect(registry):
    register(make_mock("mock-perfect"))
    yield registry.get("mock-perfect")
    unregister("mock-perfect")


def test_perfect_model_scores_high(perfect):
    manifest = run_probes(perfect, budget_usd=5.0, max_context_tokens=4_000)
    assert manifest.json_reliability == 1.0
    assert manifest.instruction_score == 1.0
    assert manifest.tool_use_reliability == 1.0
    assert manifest.usable_context_tokens == 4_000
    assert manifest.tokens_per_second > 0
    assert manifest.probe_cost_usd > 0
    # persisted and reloadable
    assert CapabilityManifest.load("mock-perfect").json_reliability == 1.0


def test_broken_model_scores_zero(registry):
    register(make_mock("mock-broken", responder=broken_responder))
    try:
        manifest = run_probes(registry.get("mock-broken"), budget_usd=5.0,
                              skip_context=True)
        assert manifest.json_reliability == 0.0
        assert manifest.instruction_score == 0.0
        assert manifest.tool_use_reliability == 0.0
    finally:
        unregister("mock-broken")


def test_budget_guard_stops_probing(registry):
    expensive = make_mock("mock-expensive",
                          pricing=Pricing(1_000_000.0, 1_000_000.0))
    register(expensive)
    try:
        with pytest.raises(BudgetExceededError):
            run_probes(registry.get("mock-expensive"), budget_usd=0.01)
        # partial manifest still saved
        manifest = CapabilityManifest.load("mock-expensive")
        assert manifest is not None
        assert "budget_exceeded" in manifest.notes
    finally:
        unregister("mock-expensive")


def test_limited_context_model_measured_correctly(registry):
    register(make_mock("mock-small-ctx", max_context_tokens=5_000))
    try:
        manifest = run_probes(registry.get("mock-small-ctx"), budget_usd=5.0,
                              max_context_tokens=16_000)
        # passes at 1k and 4k, fails at 16k -> usable context is 4k
        assert manifest.usable_context_tokens == 4_000
    finally:
        unregister("mock-small-ctx")
