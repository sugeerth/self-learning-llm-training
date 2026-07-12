from onramp import CapabilityManifest, register, unregister
from onramp.autopilot import apply, evaluate
from onramp.stats import StatsStore
from onramp.testing import make_mock


def _live(stats, model_id, ok, bad, score=None):
    for _ in range(ok):
        stats.record_call(model_id, "judge", success=True)
    for _ in range(bad):
        stats.record_call(model_id, "judge", success=False)
    if score is not None:
        stats.record_score(model_id, "judge", score)


def test_candidate_earns_promotion(tmp_path, registry):
    stats = StatsStore(tmp_path / "s.json")
    register(make_mock("ap-newcomer"))
    CapabilityManifest(model_id="ap-newcomer", json_reliability=0.99).save()
    _live(stats, "ap-newcomer", ok=30, bad=0, score=0.9)
    try:
        actions = evaluate(registry=registry, stats=stats)
        assert [(a.model_id, a.action) for a in actions] == [("ap-newcomer", "promote")]
        apply(actions)
        assert CapabilityManifest.load("ap-newcomer").status == "stable"
    finally:
        unregister("ap-newcomer")


def test_too_few_calls_blocks_promotion(tmp_path, registry):
    stats = StatsStore(tmp_path / "s.json")
    register(make_mock("ap-thin"))
    CapabilityManifest(model_id="ap-thin", json_reliability=0.99).save()
    _live(stats, "ap-thin", ok=5, bad=0)
    try:
        assert evaluate(registry=registry, stats=stats) == []
    finally:
        unregister("ap-thin")


def test_unpriced_model_never_auto_promotes(tmp_path, registry):
    stats = StatsStore(tmp_path / "s.json")
    register(make_mock("ap-unpriced"))
    manifest = CapabilityManifest(model_id="ap-unpriced", json_reliability=0.99)
    manifest.notes["pricing_unknown"] = True
    manifest.save()
    _live(stats, "ap-unpriced", ok=50, bad=0, score=1.0)
    try:
        assert evaluate(registry=registry, stats=stats) == []
    finally:
        unregister("ap-unpriced")


def test_quality_below_stable_cohort_blocks_promotion(tmp_path, registry):
    stats = StatsStore(tmp_path / "s.json")
    register(make_mock("ap-stable"))
    register(make_mock("ap-mediocre"))
    incumbent = CapabilityManifest(model_id="ap-stable", json_reliability=0.99)
    incumbent.save()
    incumbent.set_status("stable")
    CapabilityManifest(model_id="ap-mediocre", json_reliability=0.99).save()
    _live(stats, "ap-stable", ok=30, bad=0, score=0.95)
    _live(stats, "ap-mediocre", ok=30, bad=0, score=0.5)   # reliable but bad
    try:
        actions = evaluate(registry=registry, stats=stats)
        assert all(a.model_id != "ap-mediocre" for a in actions)
    finally:
        unregister("ap-stable")
        unregister("ap-mediocre")


def test_failing_stable_gets_demoted(tmp_path, registry):
    stats = StatsStore(tmp_path / "s.json")
    register(make_mock("ap-decayed"))
    manifest = CapabilityManifest(model_id="ap-decayed", json_reliability=0.99)
    manifest.save()
    manifest.set_status("stable")
    _live(stats, "ap-decayed", ok=10, bad=20)   # 33% success
    try:
        actions = evaluate(registry=registry, stats=stats)
        assert [(a.model_id, a.action) for a in actions] == [("ap-decayed", "demote")]
        apply(actions)
        assert CapabilityManifest.load("ap-decayed").status == "candidate"
    finally:
        unregister("ap-decayed")
