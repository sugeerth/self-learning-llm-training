"""Baseline vs Trajectory-Learned — the project's evaluation.

Runs both agents over the held-out suite and reports, per benchmark and
overall, the metrics the spec asks to track: success rate, cost per successful
task, runtime, and tool efficiency. Everything is deterministic and offline.
"""

from __future__ import annotations

from collections import defaultdict

from .agents import BaselineAgent, TrajectoryLearnedAgent
from .benchmark import BENCHMARKS, Task, grade, mock_suite
from .memory import TrajectoryMemory
from .trajectory import synth_dataset


def _run(agent, tasks: list[Task]) -> dict:
    per = defaultdict(lambda: {"n": 0, "solved": 0, "steps": 0,
                               "cost": 0.0, "runtime": 0.0, "tool_eff": 0.0})
    for task in tasks:
        r = agent.act(task.domain, task.goal)
        g = grade(task, r.tools)
        b = per[task.domain]
        b["n"] += 1
        b["solved"] += int(g["solved"])
        b["steps"] += g["steps"]
        b["cost"] += g["cost_usd"]
        b["runtime"] += g["runtime_s"]
        b["tool_eff"] += g["tool_efficiency"]

    report = {}
    tot = {"n": 0, "solved": 0, "cost": 0.0, "runtime": 0.0, "tool_eff": 0.0}
    for domain, b in per.items():
        n = b["n"]
        solved = b["solved"]
        report[BENCHMARKS[domain]] = {
            "success_rate": round(solved / n, 3),
            "cost_per_success": round(b["cost"] / solved, 4) if solved else None,
            "avg_runtime_s": round(b["runtime"] / n, 2),
            "tool_efficiency": round(b["tool_eff"] / n, 3),
            "n": n,
        }
        for k, kk in (("n", "n"), ("solved", "solved"), ("cost", "cost"),
                      ("runtime", "runtime"), ("tool_eff", "tool_eff")):
            tot[kk] += b[k]
    report["OVERALL"] = {
        "success_rate": round(tot["solved"] / tot["n"], 3),
        "cost_per_success": round(tot["cost"] / tot["solved"], 4) if tot["solved"] else None,
        "avg_runtime_s": round(tot["runtime"] / tot["n"], 2),
        "tool_efficiency": round(tot["tool_eff"] / tot["n"], 3),
        "n": tot["n"],
    }
    return report


def compare(n_per_family_train: int = 20, n_per_family_eval: int = 10,
            seed: int = 0, k: int = 5, min_quality: float = 0.6) -> dict:
    """Build memory from (synthetic) public trajectories, evaluate both agents,
    and return baseline vs learned with deltas."""
    trajs = synth_dataset(n_per_family=n_per_family_train, seed=seed)
    memory = TrajectoryMemory(trajs, min_quality=min_quality)
    tasks = mock_suite(n_per_family=n_per_family_eval, seed=seed + 100)

    baseline = _run(BaselineAgent(), tasks)
    learned = _run(TrajectoryLearnedAgent(memory, k=k), tasks)

    deltas = {}
    for bench in baseline:
        b, l = baseline[bench], learned[bench]
        deltas[bench] = {
            "success_rate": round(l["success_rate"] - b["success_rate"], 3),
            "tool_efficiency": round(l["tool_efficiency"] - b["tool_efficiency"], 3),
        }
    return {"memory_size": len(memory), "n_trajectories": len(trajs),
            "eval_tasks": len(tasks), "baseline": baseline,
            "trajectory_learned": learned, "delta": deltas}


def format_report(result: dict) -> str:
    lines = [f"Trajectory memory: {result['memory_size']} curated / "
             f"{result['n_trajectories']} collected | "
             f"eval tasks: {result['eval_tasks']}", ""]
    hdr = f"{'benchmark':<22}{'baseline':>10}{'learned':>10}{'Δ success':>11}{'tool_eff Δ':>12}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for bench in result["baseline"]:
        b = result["baseline"][bench]["success_rate"]
        l = result["trajectory_learned"][bench]["success_rate"]
        d = result["delta"][bench]
        star = "  *" if bench == "OVERALL" else ""
        lines.append(f"{bench:<22}{b:>10.0%}{l:>10.0%}"
                     f"{d['success_rate']:>+11.0%}{d['tool_efficiency']:>+12.0%}{star}")
    return "\n".join(lines)
