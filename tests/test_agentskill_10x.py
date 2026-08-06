"""Tests for agentskill v2 — closed-loop env, skill mining, ablations, transfer."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentskill.agents import (BaselinePolicy, LearnedPolicy, PlanPolicy,
                               ablation_policies)
from agentskill.benchmark import Task, mock_suite
from agentskill.env import Episode, ToolEnv
from agentskill.evaluate import _mean_ci, full_evaluation
from agentskill.memory import TrajectoryMemory
from agentskill.mining import mine_recovery, mine_skills
from agentskill.trajectory import Step, Trajectory, synth_dataset


def _task(domain="gaia", goal="Find a book's author (t)"):
    return Task(task_id=f"t-{domain}", domain=domain, goal=goal)


class ScriptedPolicy:
    """Plays a fixed tool list, ignoring outcomes (for env unit tests)."""

    def __init__(self, tools):
        self.tools = list(tools)

    def reset(self, domain, goal):
        self.i = 0

    def next_tool(self, history):
        if self.i >= len(self.tools):
            return None
        t = self.tools[self.i]
        self.i += 1
        return t


class TestToolEnv(unittest.TestCase):
    def test_deterministic_per_seed(self):
        task = _task()
        p = ScriptedPolicy(task.canonical)
        a = ToolEnv(task, fail_rate=0.5, seed=3).run(p)
        b = ToolEnv(task, fail_rate=0.5, seed=3).run(p)
        self.assertEqual(a.steps, b.steps)

    def test_zero_fail_rate_solves(self):
        task = _task()
        ep = ToolEnv(task, fail_rate=0.0, seed=0).run(ScriptedPolicy(task.canonical))
        self.assertTrue(ep.solved)
        self.assertTrue(all(ok for _, ok in ep.steps))

    def test_immediate_retry_always_succeeds(self):
        task = _task()
        # play each canonical tool twice: any failure is followed by the same
        # tool, which the env guarantees to succeed
        doubled = [t for t in task.canonical for _ in range(2)]
        ep = ToolEnv(task, fail_rate=0.9, seed=1).run(ScriptedPolicy(doubled))
        for i, (tool, ok) in enumerate(ep.steps):
            if not ok and i + 1 < len(ep.steps) and ep.steps[i + 1][0] == tool:
                self.assertTrue(ep.steps[i + 1][1])
        self.assertTrue(ep.solved)

    def test_max_steps_bounds_episode(self):
        task = _task()
        ep = ToolEnv(task, fail_rate=0.0, seed=0,
                     max_steps=2).run(ScriptedPolicy(task.canonical * 5))
        self.assertEqual(len(ep.steps), 2)

    def test_solved_needs_ok_subsequence(self):
        task = _task()
        ep = Episode(task=task, steps=[(t, True) for t in task.canonical])
        self.assertTrue(ep.solved)
        # same tools but one canonical step failed -> not solved
        broken = [(t, True) for t in task.canonical]
        broken[1] = (broken[1][0], False)
        self.assertFalse(Episode(task=task, steps=broken).solved)


class TestMining(unittest.TestCase):
    def _traj(self, steps, domain="swe", success=True, i=0):
        return Trajectory(task_id=f"m-{i}", domain=domain, goal="g",
                          steps=steps, success=success)

    def test_recovery_learned_from_data(self):
        trajs = [self._traj([Step("x", "edit", "err", ok=False),
                             Step("x", "edit", "ok", ok=True)], i=i)
                 for i in range(6)]
        rule = mine_recovery(trajs)
        self.assertTrue(rule.retry)
        self.assertEqual(rule.p_success, 1.0)
        self.assertEqual(rule.support, 6)

    def test_recovery_off_without_support(self):
        trajs = [self._traj([Step("x", "edit", "err", ok=False),
                             Step("x", "edit", "ok", ok=True)], i=i)
                 for i in range(2)]           # only 2 events < min_support
        self.assertFalse(mine_recovery(trajs).retry)

    def test_recovery_off_when_retries_fail(self):
        trajs = [self._traj([Step("x", "edit", "err", ok=False),
                             Step("x", "edit", "err again", ok=False)], i=i)
                 for i in range(10)]
        rule = mine_recovery(trajs)
        self.assertFalse(rule.retry)
        self.assertEqual(rule.p_success, 0.0)

    def test_synth_dataset_supports_recovery_rule(self):
        rule = mine_recovery(synth_dataset(n_per_family=20, seed=0))
        self.assertTrue(rule.retry)
        self.assertGreaterEqual(rule.support, 5)

    def test_mine_skills_bigrams(self):
        trajs = [self._traj([Step("x", "edit", "", True),
                             Step("x", "run_tests", "", True)], i=i)
                 for i in range(4)]
        skills = mine_skills(trajs, n=2)
        self.assertEqual(skills["swe"][0], (("edit", "run_tests"), 4))

    def test_mine_skills_ignores_failures(self):
        trajs = [self._traj([Step("x", "edit", "", False),
                             Step("x", "run_tests", "", True)],
                            success=False, i=0)]
        self.assertEqual(mine_skills(trajs), {})


class TestClosedLoopPolicies(unittest.TestCase):
    def test_baseline_never_retries(self):
        p = BaselinePolicy()
        p.reset("swe", "Fix a failing unit test")
        first = p.next_tool([])
        nxt = p.next_tool([(first, False)])     # failure ignored -> moves on
        self.assertNotEqual(nxt, first)

    def test_recovery_policy_retries_once(self):
        rule = mine_recovery(synth_dataset(seed=0))
        p = PlanPolicy(recovery=rule)
        p.reset("swe", "Fix a failing unit test")
        first = p.next_tool([])
        self.assertEqual(p.next_tool([(first, False)]), first)   # retry
        # after a successful retry, continue with the plan
        follow = p.next_tool([(first, False), (first, True)])
        self.assertNotEqual(follow, first)

    def test_retry_budget_is_bounded(self):
        rule = mine_recovery(synth_dataset(seed=0))
        p = PlanPolicy(recovery=rule)
        p.reset("swe", "Fix a failing unit test")
        first = p.next_tool([])
        self.assertEqual(p.next_tool([(first, False)]), first)
        # second consecutive failure of the same tool: budget spent, move on
        nxt = p.next_tool([(first, False), (first, False)])
        self.assertNotEqual(nxt, first)

    def test_learned_policy_adapts_plan_to_family(self):
        memory = TrajectoryMemory(synth_dataset(seed=0))
        p = LearnedPolicy(memory)
        p.reset("gaia", "Convert units and answer (x)")
        self.assertIn("calculator", p.plan)
        p.reset("gaia", "Find a book's author (x)")
        self.assertNotIn("calculator", p.plan)

    def test_ablation_grid_shape(self):
        memory = TrajectoryMemory(synth_dataset(seed=0))
        rule = mine_recovery(synth_dataset(seed=0))
        grid = ablation_policies(memory, rule)
        self.assertEqual(set(grid),
                         {"baseline", "+plans", "+recovery", "learned (full)"})
        self.assertIsNone(grid["baseline"].recovery)
        self.assertIsNone(grid["+plans"].recovery)
        self.assertIs(grid["+recovery"].recovery, rule)
        self.assertIs(grid["learned (full)"].recovery, rule)


class TestFullEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = full_evaluation(seeds=(0, 1), n_per_family_train=20,
                                  n_per_family_eval=4, fail_rate=0.15)

    def test_ablation_ordering(self):
        sr = {n: v["success_rate"]["mean"]
              for n, v in self.res["ablations"].items()}
        self.assertGreater(sr["learned (full)"], sr["baseline"])
        self.assertGreaterEqual(sr["learned (full)"], sr["+plans"])
        self.assertGreaterEqual(sr["learned (full)"], sr["+recovery"])
        self.assertGreaterEqual(sr["+plans"], sr["baseline"])
        self.assertGreaterEqual(sr["+recovery"], sr["baseline"])

    def test_recovery_rule_mined_not_hardcoded(self):
        rr = self.res["recovery_rule"]
        self.assertTrue(rr["retry"])
        self.assertGreaterEqual(rr["support"], 5)
        self.assertGreaterEqual(rr["p_success"], 0.7)

    def test_transfer_recovery_survives_lodo(self):
        for held, v in self.res["transfer"].items():
            self.assertGreaterEqual(v["learned"]["mean"], v["baseline"]["mean"],
                                    f"LODO lift missing on {held}")
        # and at least one domain shows a strict lift
        self.assertTrue(any(v["learned"]["mean"] > v["baseline"]["mean"]
                            for v in self.res["transfer"].values()))

    def test_ci_fields_present(self):
        for cfg in self.res["ablations"].values():
            for metric in ("success_rate", "tool_efficiency", "avg_steps"):
                self.assertIn("mean", cfg[metric])
                self.assertIn("ci95", cfg[metric])

    def test_skills_cover_all_domains(self):
        self.assertEqual(set(self.res["skills"]), {"gaia", "mle", "swe"})


class TestMeanCI(unittest.TestCase):
    def test_single_value_has_zero_ci(self):
        self.assertEqual(_mean_ci([0.5]), (0.5, 0.0))

    def test_constant_sample_has_zero_ci(self):
        m, ci = _mean_ci([0.8, 0.8, 0.8])
        self.assertAlmostEqual(m, 0.8)
        self.assertAlmostEqual(ci, 0.0)

    def test_ci_positive_for_varied_sample(self):
        m, ci = _mean_ci([0.4, 0.6])
        self.assertAlmostEqual(m, 0.5)
        self.assertGreater(ci, 0.0)


class TestEndToEnd(unittest.TestCase):
    def test_learned_beats_baseline_in_noisy_env(self):
        trajs = synth_dataset(n_per_family=20, seed=0)
        memory = TrajectoryMemory(trajs)
        rule = mine_recovery(trajs)
        tasks = mock_suite(n_per_family=4, seed=777)
        base_solved = learned_solved = 0
        for task in tasks:
            base_solved += int(ToolEnv(task, fail_rate=0.15, seed=0)
                               .run(BaselinePolicy()).solved)
            learned_solved += int(ToolEnv(task, fail_rate=0.15, seed=0)
                                  .run(LearnedPolicy(memory, recovery=rule))
                                  .solved)
        self.assertGreater(learned_solved, base_solved)


if __name__ == "__main__":
    unittest.main()
