from onramp import CapabilityManifest, detect_drift


def test_no_drift_with_single_snapshot():
    CapabilityManifest(model_id="m", json_reliability=0.98,
                       probed_at="2026-07-01T00:00:00+00:00").save()
    assert detect_drift("m") == []


def test_drift_detected_on_regression():
    CapabilityManifest(model_id="m", json_reliability=0.98, tokens_per_second=100,
                       probed_at="2026-07-01T00:00:00+00:00").save()
    CapabilityManifest(model_id="m", json_reliability=0.70, tokens_per_second=101,
                       probed_at="2026-07-08T00:00:00+00:00").save()
    alerts = detect_drift("m", threshold=0.10)
    assert alerts == ["json_reliability: 0.98 -> 0.7"]  # tps change is < 10%


def test_stable_snapshots_report_clean():
    CapabilityManifest(model_id="m", json_reliability=0.98,
                       probed_at="2026-07-01T00:00:00+00:00").save()
    CapabilityManifest(model_id="m", json_reliability=0.96,
                       probed_at="2026-07-08T00:00:00+00:00").save()
    assert detect_drift("m", threshold=0.10) == []
