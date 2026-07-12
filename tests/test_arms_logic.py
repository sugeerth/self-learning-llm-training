"""Tests for the arms framework's accounting (needs torch to import arms)."""
import pytest

pytest.importorskip("torch")

from arms import Trajectory, _bracket_cost, make_report
from hyperband import Bracket


def _traj(points):
    """points: list of (delta_steps, ppl)."""
    t = Trajectory()
    for delta, ppl in points:
        t.record({"val_ppl": ppl}, delta)
    return t


def test_trajectory_best_is_monotone_and_indexed_by_spend():
    t = _traj([(10, 100.0), (10, 80.0), (10, 90.0), (10, 75.0)])
    assert t.spent == 40
    assert [p["best"] for p in t.points] == [100.0, 80.0, 80.0, 75.0]
    assert t.best_at(20) == 80.0
    assert t.best_at(5) == float("inf")     # nothing finished yet
    assert t.steps_to_reach(80.0) == 20
    assert t.steps_to_reach(1.0) is None


def test_bracket_cost_counts_promoted_deltas_not_from_scratch():
    b = Bracket(n_candidates=4, halvings=2, initial_steps=4, eta=2)
    # rung0: 4x4, rung1: 2 survivors x (8-4), rung2: 1 x (16-8)
    assert _bracket_cost(b) == 16 + 8 + 8
    # from-scratch would be 4x4 + 2x8 + 1x16 = 48 — promotion must be cheaper
    assert _bracket_cost(b) < 48


def test_make_report_normalizes_speedup_by_randoms_own_reach():
    budget = 100
    # random finds its final best (50.0) after 60 steps
    rand = _traj([(20, 80.0), (20, 60.0), (20, 50.0), (20, 55.0), (20, 52.0)])
    # smart arm reaches 50.0 quality after 30 steps
    smart = _traj([(10, 70.0), (10, 55.0), (10, 49.0)])
    report = make_report({"random": [rand], "smart": [smart]}, budget, skipped=[])

    assert report["regret_target_ppl"] == 50.0
    assert report["random_steps_to_target"] == 60
    # random against itself is ~1x by construction, not budget/steps
    assert report["arms"]["random"]["speedup_vs_random"] == pytest.approx(1.0)
    assert report["arms"]["smart"]["speedup_vs_random"] == pytest.approx(2.0)
    assert report["arms"]["smart"]["steps_to_random_final"] == 30
