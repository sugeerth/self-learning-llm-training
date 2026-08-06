"""Learning general agent skills from trajectories — offline, deterministic."""

from agentskill.agents import BaselineAgent, TrajectoryLearnedAgent
from agentskill.benchmark import Task, grade, mock_suite
from agentskill.evaluate import compare
from agentskill.finetune import build_sft_dataset, to_sft_example
from agentskill.memory import TrajectoryMemory
from agentskill.scoring import curate, score_trajectory
from agentskill.trajectory import (Trajectory, canonical_for, family_of,
                                   synth_dataset, synth_trajectory)
import random


# ── schema / families ────────────────────────────────────────────────────

def test_family_inferred_from_goal():
    assert family_of("gaia", "Convert a currency amount") == "convert"
    assert family_of("gaia", "Find the population") == "lookup"
    assert family_of("mle", "Tune the model") == "tuned"
    assert family_of("swe", "Debug a flaky test") == "debug"
    assert family_of("swe", "Fix a bug") == "fix"


def test_canonical_differs_by_family():
    assert "calculator" in canonical_for("gaia", "convert units")
    assert "calculator" not in canonical_for("gaia", "find a date")


def test_trajectory_roundtrip_and_recovery():
    t = synth_trajectory("swe", "fix", 0, random.Random(1))
    t2 = Trajectory.from_dict(t.to_dict())
    assert t2.tools == t.tools and t2.success == t.success
    rec = Trajectory("x", "gaia", "g",
                     steps=[__import__("agentskill.trajectory", fromlist=["Step"]).Step("t", "browse", "dead", ok=False)],
                     success=True)
    assert rec.recovered


# ── scoring ────────────────────────────────────────────────────────────────

def test_success_scores_above_failure():
    good = synth_trajectory("gaia", "lookup", 0, random.Random(2))
    while not good.success:
        good = synth_trajectory("gaia", "lookup", 0, random.Random(3))
    bad = Trajectory("b", "gaia", "g", steps=good.steps[:1], success=False)
    assert score_trajectory(good).total > score_trajectory(bad).total


def test_curate_keeps_only_high_quality():
    trajs = synth_dataset(n_per_family=10, seed=0)
    kept = curate(trajs, min_quality=0.6)
    assert kept and all(s.total >= 0.6 for _, s in kept)
    assert all(t.success for t, _ in kept)      # failures fall below the bar


# ── memory / retrieval ─────────────────────────────────────────────────────

def test_memory_retrieves_same_domain_and_family():
    mem = TrajectoryMemory(synth_dataset(n_per_family=15, seed=0))
    hits = mem.retrieve("gaia Convert a currency amount", k=5)
    assert hits
    assert any(t.domain == "gaia" for t, _ in hits)
    assert any("calculator" in t.tools for t, _ in hits)   # convert-family found


# ── agents / benchmark ─────────────────────────────────────────────────────

def test_grade_subsequence_match():
    task = Task("t", "gaia", "Convert a currency amount")   # needs calculator
    assert grade(task, ["search", "read", "calculator", "answer"])["solved"]
    assert not grade(task, ["search", "read", "extract", "answer"])["solved"]


def test_baseline_fails_nondefault_family():
    task = Task("t", "gaia", "Convert a currency amount (eval)")
    r = BaselineAgent().act(task.domain, task.goal)
    assert not grade(task, r.tools)["solved"]   # baseline uses lookup plan


def test_learned_agent_solves_nondefault_family():
    mem = TrajectoryMemory(synth_dataset(n_per_family=20, seed=0))
    agent = TrajectoryLearnedAgent(mem, k=5)
    task = Task("t", "gaia", "Convert a currency amount (eval)")
    assert grade(task, agent.act(task.domain, task.goal).tools)["solved"]


# ── end-to-end comparison ───────────────────────────────────────────────────

def test_learned_beats_baseline_overall():
    res = compare(seed=0)
    b = res["baseline"]["OVERALL"]["success_rate"]
    l = res["trajectory_learned"]["OVERALL"]["success_rate"]
    assert l > b
    assert res["delta"]["OVERALL"]["success_rate"] > 0
    # every benchmark improves or holds; none regresses
    assert all(res["delta"][k]["success_rate"] >= 0 for k in res["delta"])


def test_cost_per_success_reported():
    res = compare(seed=0)
    for agent in ("baseline", "trajectory_learned"):
        cps = res[agent]["OVERALL"]["cost_per_success"]
        assert cps is None or cps > 0


# ── fine-tune dataset (CPU part of the GPU step) ────────────────────────────

def test_sft_example_teaches_canonical_plan():
    t = synth_trajectory("gaia", "convert", 0, random.Random(0))
    ex = to_sft_example(t)
    assert "calculator" in ex["completion"]     # family-correct plan
    assert ex["prompt"].strip().endswith("Plan:")


def test_build_sft_dataset_dedupes_and_gates():
    trajs = synth_dataset(n_per_family=10, seed=0)
    ex = build_sft_dataset(trajs, min_quality=0.6)
    prompts = [e["prompt"] for e in ex]
    assert len(prompts) == len(set(prompts))    # deduped
    assert ex
