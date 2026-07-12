import pytest

from onramp import (CapabilityManifest, Pricing, RoleProfile, Router,
                    register, unregister)
from onramp.testing import make_mock


@pytest.fixture
def cheap_candidate_vs_pricey_stable(registry):
    register(make_mock("mock-newcomer", pricing=Pricing(0.1, 0.2)))
    register(make_mock("mock-incumbent", pricing=Pricing(3.0, 6.0)))
    CapabilityManifest(model_id="mock-newcomer", json_reliability=0.99,
                       output_per_mtok=0.2).save()                 # candidate
    incumbent = CapabilityManifest(model_id="mock-incumbent",
                                   json_reliability=0.95, output_per_mtok=6.0)
    incumbent.save()
    incumbent.set_status("stable")
    yield
    unregister("mock-newcomer")
    unregister("mock-incumbent")


def _judge_router(registry):
    return Router(registry, roles={"judge": RoleProfile(
        "judge", needs={"json_reliability": 0.9})})


def test_stable_outranks_cheaper_candidate(cheap_candidate_vs_pricey_stable, registry):
    router = _judge_router(registry)
    # newcomer is cheaper AND scores higher, but hasn't been promoted yet
    assert router.candidates("judge") == ["mock-incumbent", "mock-newcomer"]


def test_promotion_flips_the_ranking(cheap_candidate_vs_pricey_stable, registry):
    CapabilityManifest.load("mock-newcomer").set_status("stable")
    router = _judge_router(registry)
    assert router.candidates("judge") == ["mock-newcomer", "mock-incumbent"]


def test_retired_model_never_routes(cheap_candidate_vs_pricey_stable, registry):
    CapabilityManifest.load("mock-incumbent").set_status("retired")
    router = _judge_router(registry)
    assert router.candidates("judge") == ["mock-newcomer"]


def test_invalid_status_rejected(cheap_candidate_vs_pricey_stable):
    with pytest.raises(ValueError):
        CapabilityManifest.load("mock-newcomer").set_status("golden")


def test_status_change_does_not_pollute_history(cheap_candidate_vs_pricey_stable):
    before = len(CapabilityManifest.history("mock-newcomer"))
    CapabilityManifest.load("mock-newcomer").set_status("stable")
    assert len(CapabilityManifest.history("mock-newcomer")) == before
