"""Unit tests for the dependency-light pure logic in this repo.

These import the REAL functions under test (no logic is copied). torch/anthropic/
flask happen to be importable in CI here, but every test below exercises only
deterministic, hardware-free logic so they run fast and offline.
"""
from __future__ import annotations

import math
import random

import pytest


# ───────────────────────── model.py: FFN hidden-dim rounding ─────────────────────────

def test_round_up_matches_legacy_inline_expression():
    """_round_up(n) must equal the old inline `64 * ((n + 63) // 64)` exactly."""
    from model import _round_up, FFN_HIDDEN_MULTIPLE

    assert FFN_HIDDEN_MULTIPLE == 64

    def legacy(n: int) -> int:
        return 64 * ((n + 63) // 64)

    rng = random.Random(0)
    samples = list(range(0, 4096)) + [rng.randint(0, 1_000_000) for _ in range(500)]
    for n in samples:
        assert _round_up(n) == legacy(n), n


def test_round_up_is_a_multiple_and_not_smaller():
    from model import _round_up

    for n in [1, 63, 64, 65, 1000, 1023, 1024, 1025]:
        r = _round_up(n)
        assert r % 64 == 0
        assert r >= n
        assert r - n < 64


def test_round_up_custom_multiple():
    from model import _round_up

    assert _round_up(10, 8) == 16
    assert _round_up(16, 8) == 16
    assert _round_up(17, 8) == 24


# ───────────────────────── hyperband.py ─────────────────────────

def test_standard_brackets_shapes():
    from hyperband import standard_brackets, Bracket

    brackets = standard_brackets(max_steps=200, eta=2)
    assert brackets, "should produce at least one bracket"
    assert all(isinstance(b, Bracket) for b in brackets)
    # every bracket must have a positive budget and candidate count
    for b in brackets:
        assert b.n_candidates >= 1
        assert b.initial_steps >= 1
        assert b.eta == 2


def test_cheap_prior_empty_returns_default_mean():
    from hyperband import CheapPrior

    p = CheapPrior()
    mean, unc = p.predict({"n_layers": 6, "d_model": 384})
    assert mean == 5.0
    assert unc == 1.0


def test_cheap_prior_predicts_near_exact_match():
    from hyperband import CheapPrior

    p = CheapPrior(length_scale=0.5)
    cfg = {"n_layers": 6, "d_model": 384, "n_heads": 6, "n_kv_heads": 2}
    p.add(cfg, val_ppl=math.e ** 3)  # log-ppl == 3.0
    mean, unc = p.predict(cfg)
    # identical config => weighted average collapses to the single stored point
    assert mean == pytest.approx(3.0, abs=1e-9)
    assert 0.0 < unc <= 1.0


def test_acquisition_is_lcb():
    from hyperband import CheapPrior

    p = CheapPrior()
    cfg = {"n_layers": 6, "d_model": 384}
    p.add(cfg, val_ppl=math.e ** 4)
    mean, unc = p.predict(cfg)
    assert p.acquisition(cfg, kappa=1.5) == pytest.approx(mean - 1.5 * unc)


def test_random_config_is_internally_consistent():
    from hyperband import random_config

    rng = random.Random(123)
    for _ in range(200):
        c = random_config(rng)
        # n_heads must divide d_model (attention head split)
        assert c["d_model"] % c["n_heads"] == 0
        # n_kv_heads is MQA(1), GQA(2), or full MHA(n_heads)
        assert c["n_kv_heads"] in (1, 2, c["n_heads"])


def test_successive_halving_keeps_best_and_increases_steps():
    from hyperband import successive_halving, Bracket

    # train_partial: lower "id" => lower (better) val_ppl, deterministic
    calls = {"steps_seen": []}

    def train_partial(cfg, steps):
        calls["steps_seen"].append(steps)
        return {"val_ppl": float(cfg["id"])}

    candidates = [{"id": i} for i in range(4)]
    bracket = Bracket(n_candidates=4, halvings=2, initial_steps=10, eta=2)
    survivors = successive_halving(candidates, train_partial, bracket)

    assert len(survivors) == 1
    assert survivors[0]["id"] == 0  # the best candidate survives
    # steps double each halving round: 10, then 20, then 40
    assert 10 in calls["steps_seen"] and 20 in calls["steps_seen"]


# ───────────────────────── experiments.py ─────────────────────────

def test_stats_mean_std_n():
    from experiments import _stats

    s = _stats([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert s["n"] == 8
    assert s["mean"] == pytest.approx(5.0)
    assert s["std"] == pytest.approx(2.0)  # population std of this classic set


def test_stats_empty_is_safe():
    from experiments import _stats

    assert _stats([]) == {"mean": 0.0, "std": 0.0, "n": 0}


def test_lr_local_schedule_warmup_peak_and_floor():
    from experiments import lr_local_schedule

    base = 3e-4
    total = 200
    # step 0 is in warmup => strictly below base lr and positive
    assert 0 < lr_local_schedule(0, base, total) < base
    # at/after total_steps the schedule floors at 10% of base
    assert lr_local_schedule(total, base, total) == pytest.approx(base * 0.1)
    assert lr_local_schedule(total + 50, base, total) == pytest.approx(base * 0.1)
    # schedule never exceeds the base lr
    assert all(lr_local_schedule(s, base, total) <= base + 1e-12 for s in range(total))


# ───────────────────────── reasoning_pipeline.py ─────────────────────────

def test_parse_reflection_extracts_think_and_answer():
    from reasoning_pipeline import parse_reflection

    text = "<think>\nfirst I add 2+2\n</think>\n<answer>4</answer>"
    d = parse_reflection(text)
    assert d["think"] == "first I add 2+2"
    assert d["answer"] == "4"
    assert d["raw"] == text


def test_parse_reflection_missing_tags_returns_empty():
    from reasoning_pipeline import parse_reflection

    d = parse_reflection("no tags here")
    assert d["think"] == ""
    assert d["answer"] == ""


def test_math_answer_reward_correct_and_incorrect():
    from reasoning_pipeline import VerifiableReward

    gen = "<think>...</think><answer>42</answer>"
    assert VerifiableReward.math_answer(gen, "42") == 1.0
    assert VerifiableReward.math_answer(gen, "43") == 0.0


def test_math_answer_reward_multi_digit_regression():
    """Bug fix: numbers >= 1000 used to be mis-parsed (e.g. '1234' -> last frag '4').

    GSM8K answers are routinely 4+ digits, so this is the core verifiable-reward path.
    """
    from reasoning_pipeline import VerifiableReward

    assert VerifiableReward.math_answer("<answer>1234</answer>", "1234") == 1.0
    assert VerifiableReward.math_answer("<answer>1000000</answer>", "1000000") == 1.0
    # currency + thousands separators must also resolve to the full value
    assert VerifiableReward.math_answer("<answer>$1,234</answer>", "1234") == 1.0
    assert VerifiableReward.math_answer("<answer>1,234</answer>", "1234") == 1.0
    # a genuinely wrong large answer is still rejected
    assert VerifiableReward.math_answer("<answer>1235</answer>", "1234") == 0.0


def test_format_compliance_full_credit_when_ordered():
    from reasoning_pipeline import VerifiableReward

    good = "<think>reason</think><answer>x</answer>"
    assert VerifiableReward.format_compliance(good) == pytest.approx(1.0)
    # no tags at all => zero
    assert VerifiableReward.format_compliance("plain") == pytest.approx(0.0)


def test_split_into_steps_numbered_and_fallback():
    from reasoning_pipeline import ProcessRewardModel

    numbered = "\nStep 1: do a\nStep 2: do b\nStep 3: done"
    steps = ProcessRewardModel.split_into_steps(numbered)
    assert [s.text for s in steps] == ["do a", "do b", "done"]
    # step_idx tracks the position in the raw split (the leading pre-"Step 1" chunk
    # occupies index 0 and is then filtered out), so real steps start at 1 here.
    assert [s.step_idx for s in steps] == [1, 2, 3]

    para = "para one\n\npara two"
    fallback = ProcessRewardModel.split_into_steps(para)
    assert [s.text for s in fallback] == ["para one", "para two"]


def test_mcts_node_value_and_trace():
    from reasoning_pipeline import MCTSNode

    root = MCTSNode(step_text="root")
    child = MCTSNode(step_text="child", parent=root)
    root.children.append(child)

    child.visits = 4
    child.value_sum = 2.0
    assert child.avg_value == pytest.approx(0.5)

    # avg_value of an unvisited node must not divide by zero
    fresh = MCTSNode(step_text="fresh")
    assert fresh.avg_value == 0.0

    # trace runs root -> child in order
    assert [n.step_text for n in child.trace()] == ["root", "child"]
    # root (no parent) ucb == its own avg_value
    assert root.ucb() == pytest.approx(root.avg_value)


# ───────────────────────── agents.py: JSON extraction ─────────────────────────

def test_extract_json_from_fenced_block():
    from agents import BaseAgent

    text = 'sure!\n```json\n{"name": "x", "lr": 0.0003}\n```\nthanks'
    assert BaseAgent._extract_json(text) == {"name": "x", "lr": 0.0003}


def test_extract_json_bare_braces():
    from agents import BaseAgent

    text = 'prefix {"accept": true, "confidence": 0.9} suffix'
    assert BaseAgent._extract_json(text) == {"accept": True, "confidence": 0.9}
