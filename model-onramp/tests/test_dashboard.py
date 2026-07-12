import json

from onramp import CapabilityManifest, register, unregister
from onramp.dashboard import build_state, export_feed


def test_build_state_and_feed(tmp_path, registry):
    from onramp.testing import make_mock

    register(make_mock("mock-dash"))
    CapabilityManifest(model_id="mock-dash", json_reliability=0.9,
                       probed_at="2026-07-01T00:00:00+00:00").save()
    try:
        state = build_state()
        by_id = {m["model_id"]: m for m in state["models"]}
        assert by_id["mock-dash"]["probed"] is True
        assert by_id["mock-dash"]["manifest"]["json_reliability"] == 0.9
        assert by_id["mock-dash"]["snapshots"] == 1
        assert by_id["claude-opus-4-8"]["probed"] is False

        feed = export_feed(tmp_path / "feed.json")
        parsed = json.loads(feed.read_text())
        assert any(m["model_id"] == "mock-dash" for m in parsed["models"])
    finally:
        unregister("mock-dash")
