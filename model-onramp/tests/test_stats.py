from onramp.stats import StatsStore


def test_record_and_rates(tmp_path):
    stats = StatsStore(tmp_path / "s.json")
    for _ in range(9):
        stats.record_call("m", "judge", success=True, cost_usd=0.01)
    stats.record_call("m", "judge", success=False)
    assert stats.calls("m") == 10                       # aggregate
    assert stats.calls("m", "judge") == 10              # per-role
    assert stats.calls("m", "trainer") == 0
    # smoothed: (9+1)/(10+2)
    assert abs(stats.success_rate("m") - 10 / 12) < 1e-9
    assert abs(stats.get("m")["cost_usd"] - 0.09) < 1e-9


def test_scores(tmp_path):
    stats = StatsStore(tmp_path / "s.json")
    assert stats.mean_score("m") is None
    stats.record_score("m", "judge", 0.8)
    stats.record_score("m", "judge", 0.6)
    stats.record_score("m", None, 5.0)  # clamped to 1.0
    assert abs(stats.mean_score("m") - (0.8 + 0.6 + 1.0) / 3) < 1e-9
    assert abs(stats.mean_score("m", "judge") - 0.7) < 1e-9


def test_persistence(tmp_path):
    path = tmp_path / "s.json"
    StatsStore(path).record_call("m", None, success=True)
    assert StatsStore(path).calls("m") == 1


def test_breaker_opens_and_cools_down(tmp_path):
    stats = StatsStore(tmp_path / "s.json")
    t = 1000.0
    for _ in range(2):
        stats.record_call("m", None, success=False, now=t)
    assert not stats.breaker_open("m", now=t)           # below threshold
    stats.record_call("m", None, success=False, now=t)
    assert stats.breaker_open("m", now=t)               # 3rd trip -> open
    assert stats.breaker_open("m", now=t + 59)          # still cooling
    assert not stats.breaker_open("m", now=t + 61)      # half-open
    stats.record_call("m", None, success=True, now=t + 61)
    assert not stats.breaker_open("m", now=t + 61)      # success resets
