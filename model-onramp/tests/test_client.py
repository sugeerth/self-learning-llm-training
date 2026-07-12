import pytest

from onramp import (AllCandidatesFailedError, BudgetExceededError,
                    CapabilityManifest, OnrampClient, Pricing, RoleProfile,
                    Router, register, unregister)
from onramp.testing import MockAdapter, make_mock, perfect_responder


@pytest.fixture
def failover_setup(registry):
    register(make_mock("mock-down", pricing=Pricing(0.1, 0.1),
                       fail_with=ConnectionError("provider outage")))
    register(make_mock("mock-up", pricing=Pricing(1.0, 2.0)))
    CapabilityManifest(model_id="mock-down", json_reliability=1.0,
                       output_per_mtok=0.1).save()
    CapabilityManifest(model_id="mock-up", json_reliability=1.0,
                       output_per_mtok=2.0).save()
    router = Router(registry, roles={"judge": RoleProfile(
        "judge", needs={"json_reliability": 0.9})})
    yield router
    unregister("mock-down")
    unregister("mock-up")


def test_failover_to_next_candidate(failover_setup):
    client = OnrampClient(router=failover_setup, max_retries=0)
    # mock-down is cheapest so it's tried first, fails, and mock-up serves
    result = client.generate("Return ONLY a JSON object", role="judge")
    assert result.model_id == "mock-up"
    assert client.spent_usd > 0


def test_retry_recovers_transient_failure(registry):
    class FlakyAdapter(MockAdapter):
        model_id = "mock-flaky"
        pricing = Pricing(1.0, 1.0)
        responder = staticmethod(perfect_responder)
        fail_with = None
        max_context_tokens = None

        def _complete(self, messages, max_tokens, temperature):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transient blip")
            return super()._complete(messages, max_tokens, temperature)

    register(FlakyAdapter)
    CapabilityManifest(model_id="mock-flaky", json_reliability=1.0).save()
    router = Router(registry, roles={"judge": RoleProfile(
        "judge", needs={"json_reliability": 0.9})})
    try:
        client = OnrampClient(router=router, max_retries=1,
                              retry_base_delay=0.01)
        result = client.generate("hello", role="judge")
        # same model served after one retry — no failover happened
        assert result.model_id == "mock-flaky"
    finally:
        unregister("mock-flaky")


def test_all_candidates_failing_raises(registry):
    register(make_mock("mock-only-down", fail_with=RuntimeError("boom")))
    CapabilityManifest(model_id="mock-only-down", json_reliability=1.0).save()
    router = Router(registry, roles={"judge": RoleProfile(
        "judge", needs={"json_reliability": 0.9})})
    try:
        with pytest.raises(AllCandidatesFailedError, match="mock-only-down"):
            OnrampClient(router=router, max_retries=0).generate("hi", role="judge")
    finally:
        unregister("mock-only-down")


def test_pinned_model_bypasses_routing(failover_setup):
    client = OnrampClient(router=failover_setup, max_retries=0)
    result = client.generate("hello", model_id="mock-up")
    assert result.model_id == "mock-up"


def test_session_cost_cap(failover_setup):
    client = OnrampClient(router=failover_setup, cost_cap_usd=1e-12,
                          max_retries=0)
    with pytest.raises(BudgetExceededError):
        client.generate("hello", role="judge")


def test_role_xor_model_id_required(failover_setup):
    client = OnrampClient(router=failover_setup)
    with pytest.raises(ValueError):
        client.generate("hello")
    with pytest.raises(ValueError):
        client.generate("hello", role="judge", model_id="mock-up")
