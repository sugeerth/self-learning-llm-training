import pytest

from onramp import (CapabilityManifest, NoEligibleModelError, Pricing,
                    RoleProfile, Router, register, unregister)
from onramp.testing import make_mock


@pytest.fixture
def two_probed_mocks(registry):
    register(make_mock("mock-cheap", pricing=Pricing(0.5, 1.0)))
    register(make_mock("mock-fast", pricing=Pricing(5.0, 10.0)))
    CapabilityManifest(model_id="mock-cheap", json_reliability=0.95,
                       instruction_score=0.9, output_per_mtok=1.0,
                       tokens_per_second=20.0).save()
    CapabilityManifest(model_id="mock-fast", json_reliability=0.95,
                       instruction_score=0.9, output_per_mtok=10.0,
                       tokens_per_second=200.0).save()
    yield
    unregister("mock-cheap")
    unregister("mock-fast")


def test_cost_ranking(two_probed_mocks, registry):
    router = Router(registry, roles={"judge": RoleProfile(
        "judge", needs={"json_reliability": 0.9})})
    assert router.candidates("judge") == ["mock-cheap", "mock-fast"]
    assert router.resolve("judge") == "mock-cheap"


def test_speed_ranking(two_probed_mocks, registry):
    router = Router(registry, roles={"drafter": RoleProfile(
        "drafter", needs={"json_reliability": 0.9}, prefer="speed")})
    assert router.candidates("drafter") == ["mock-fast", "mock-cheap"]


def test_unsatisfiable_role_raises(two_probed_mocks, registry):
    router = Router(registry, roles={"impossible": RoleProfile(
        "impossible", needs={"json_reliability": 0.999})})
    with pytest.raises(NoEligibleModelError):
        router.resolve("impossible")


def test_newly_probed_model_becomes_eligible(two_probed_mocks, registry):
    router = Router(registry, roles={"judge": RoleProfile(
        "judge", needs={"json_reliability": 0.9})})
    register(make_mock("mock-new", pricing=Pricing(0.1, 0.2)))
    try:
        assert "mock-new" not in router.candidates("judge")
        CapabilityManifest(model_id="mock-new", json_reliability=0.99,
                           output_per_mtok=0.2).save()
        # zero code changes: the new model is now first in the chain
        assert router.resolve("judge") == "mock-new"
    finally:
        unregister("mock-new")
