"""Tests for the self-learning objective (regret vs random)."""
import math

from self_learning_objective import (
    check_comparable, confidence_warnings, score_report, gate, DEFAULT_MARGIN,
)


def _report(speedups: dict[str, float | None], seeds: int = 5,
            deterministic: bool = True, budget: int = 256) -> dict:
    """Build a minimal arms report with the given per-arm speedups."""
    arms = {"random": {"seeds": seeds, "speedup_vs_random": 1.0}}
    for arm, sp in speedups.items():
        arms[arm] = {"seeds": seeds, "speedup_vs_random": sp}
    return {"budget_steps": budget, "deterministic_val": deterministic, "arms": arms}


# ─────────────────────── invariants ───────────────────────

def test_comparable_passes_on_clean_report():
    assert check_comparable(_report({"evolve": 1.5})) == []


def test_comparable_flags_nondeterministic_eval():
    v = check_comparable(_report({"evolve": 1.5}, deterministic=False))
    assert any("deterministic" in x for x in v)


def test_comparable_flags_missing_random_baseline():
    rep = _report({"evolve": 1.5})
    del rep["arms"]["random"]
    v = check_comparable(rep)
    assert any("baseline" in x for x in v)


def test_comparable_flags_unequal_seeds():
    rep = _report({"evolve": 1.5})
    rep["arms"]["evolve"]["seeds"] = 3  # random still 5
    v = check_comparable(rep)
    assert any("unequal seed" in x for x in v)


def test_too_few_seeds_is_soft_warning_not_hard_violation():
    # a thin run stays comparable (usable) but is flagged low-confidence
    rep = _report({"evolve": 1.5}, seeds=2)
    assert check_comparable(rep) == []
    assert any("seed" in x for x in confidence_warnings(rep))


def test_score_appends_low_confidence_note_for_thin_runs():
    s = score_report(_report({"evolve": 1.5}, seeds=2))
    assert s.verdict == "learning"          # still usable
    assert "low confidence" in s.rationale  # but flagged


# ─────────────────────── scoring ───────────────────────

def test_score_picks_best_learning_arm():
    s = score_report(_report({"prior": 1.3, "evolve": 1.8, "agent": None}))
    assert s.best_arm == "evolve"
    assert s.value == 1.8
    assert s.verdict == "learning"


def test_score_no_op_when_near_random():
    s = score_report(_report({"evolve": 1.0}))
    assert s.verdict == "no-op"


def test_score_regressing_when_slower_than_random():
    s = score_report(_report({"prior": 0.6}))
    assert s.verdict == "regressing"


def test_score_invalid_when_not_comparable():
    s = score_report(_report({"evolve": 1.8}, deterministic=False))
    assert s.verdict == "invalid"
    assert math.isnan(s.value)


def test_score_invalid_when_no_learner_reached_target():
    s = score_report(_report({"prior": None, "evolve": None}))
    assert s.verdict == "invalid"


# ─────────────────────── the gate ───────────────────────

def test_gate_passes_above_floor():
    g = gate(_report({"evolve": 1.5}), min_speedup=1.0)
    assert g.passed


def test_gate_fails_below_floor():
    g = gate(_report({"evolve": 0.8}), min_speedup=1.0)
    assert not g.passed
    assert any("below floor" in r for r in g.reasons)


def test_gate_fails_on_invalid_report():
    g = gate(_report({"evolve": 2.0}, deterministic=False))
    assert not g.passed


def test_gate_requires_beating_incumbent_by_margin():
    # candidate J=1.5, incumbent J=1.5 -> must beat by margin, so fails
    g = gate(_report({"evolve": 1.5}), min_speedup=1.0, incumbent=1.5)
    assert not g.passed
    # candidate clearly better than incumbent -> passes
    g2 = gate(_report({"evolve": 1.5 + DEFAULT_MARGIN + 0.01}),
              min_speedup=1.0, incumbent=1.5)
    assert g2.passed
