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


# ──────────────── v2: closed-loop, ablated, multi-seed, transfer ────────────────

_T95 = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
        8: 2.365, 9: 2.306, 10: 2.262}


def _mean_ci(values: list[float]) -> tuple[float, float]:
    """(mean, 95% CI half-width) — Student t on the seed sample."""
    import statistics
    n = len(values)
    m = statistics.fmean(values)
    if n < 2:
        return m, 0.0
    s = statistics.stdev(values)
    return m, _T95.get(n, 2.0) * s / (n ** 0.5)


def _run_policy(policy, tasks, fail_rate: float, seed: int) -> dict:
    """One policy over one suite in the noisy env -> aggregate metrics."""
    from .env import ToolEnv
    solved = steps = 0
    cost = tool_eff = 0.0
    per_domain: dict[str, list[int]] = defaultdict(list)
    for task in tasks:
        ep = ToolEnv(task, fail_rate=fail_rate, seed=seed).run(policy)
        m = ep.metrics()
        solved += int(m["solved"])
        steps += m["steps"]
        cost += m["cost_usd"]
        tool_eff += m["tool_efficiency"]
        per_domain[task.domain].append(int(m["solved"]))
    n = len(tasks)
    return {
        "success_rate": solved / n,
        "cost_per_success": (cost / solved) if solved else None,
        "avg_steps": steps / n,
        "tool_efficiency": tool_eff / n,
        "per_domain": {d: sum(v) / len(v) for d, v in per_domain.items()},
    }


def full_evaluation(seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
                    fail_rate: float = 0.15, n_per_family_train: int = 20,
                    n_per_family_eval: int = 8, k: int = 5,
                    min_quality: float = 0.6) -> dict:
    """The v2 study: closed-loop noisy env, ablation grid, multi-seed CIs,
    and a leave-one-domain-out transfer matrix. Deterministic per seed set."""
    from .agents import ablation_policies, LearnedPolicy, BaselinePolicy
    from .mining import mine_recovery, mine_skills
    from .memory import TrajectoryMemory
    from .trajectory import synth_dataset

    per_config: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    transfer: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    recovery_rule = None
    skills = None

    for seed in seeds:
        trajs = synth_dataset(n_per_family=n_per_family_train, seed=seed)
        memory = TrajectoryMemory(trajs, min_quality=min_quality)
        curated = [t for t, _ in memory.entries]
        recovery_rule = mine_recovery(curated)
        skills = skills or mine_skills(curated)
        tasks = mock_suite(n_per_family=n_per_family_eval, seed=seed + 500)

        for name, policy in ablation_policies(memory, recovery_rule, k=k).items():
            r = _run_policy(policy, tasks, fail_rate, seed)
            per_config[name]["success_rate"].append(r["success_rate"])
            per_config[name]["tool_efficiency"].append(r["tool_efficiency"])
            per_config[name]["avg_steps"].append(r["avg_steps"])
            if r["cost_per_success"] is not None:
                per_config[name]["cost_per_success"].append(r["cost_per_success"])

        # leave-one-domain-out: memory (and mined recovery) built WITHOUT the
        # held-out domain; evaluate on that domain only. Planning knowledge is
        # domain-specific and should vanish; the retry rule is procedural and
        # should transfer.
        for held in BENCHMARKS:
            lodo_trajs = [t for t in trajs if t.domain != held]
            lodo_mem = TrajectoryMemory(lodo_trajs, min_quality=min_quality)
            lodo_rule = mine_recovery([t for t, _ in lodo_mem.entries])
            held_tasks = [t for t in tasks if t.domain == held]
            base = _run_policy(BaselinePolicy(), held_tasks, fail_rate, seed)
            learned = _run_policy(LearnedPolicy(lodo_mem, recovery=lodo_rule, k=k),
                                  held_tasks, fail_rate, seed)
            transfer[held]["baseline"].append(base["success_rate"])
            transfer[held]["learned"].append(learned["success_rate"])

    def _summ(vals_by_metric):
        out = {}
        for metric, vals in vals_by_metric.items():
            m, ci = _mean_ci(vals)
            out[metric] = {"mean": round(m, 3), "ci95": round(ci, 3)}
        return out

    return {
        "seeds": list(seeds), "fail_rate": fail_rate,
        "recovery_rule": recovery_rule.as_dict() if recovery_rule else None,
        "skills": {d: [{"gram": list(g), "count": c} for g, c in tops]
                   for d, tops in (skills or {}).items()},
        "ablations": {name: _summ(v) for name, v in per_config.items()},
        "transfer": {held: {"baseline": _summ({"s": v["baseline"]})["s"],
                            "learned": _summ({"s": v["learned"]})["s"]}
                     for held, v in transfer.items()},
    }


def format_full_report(res: dict) -> str:
    L = []
    L.append("# agentskill — closed-loop evaluation report")
    L.append("")
    L.append(f"Seeds: {res['seeds']} · transient tool fail rate: "
             f"{res['fail_rate']:.0%} · every number is mean ± 95% CI over seeds.")
    L.append("")
    rr = res["recovery_rule"]
    L.append(f"**Mined recovery rule** (from trajectories, not hardcoded): "
             f"retry={rr['retry']}, P(retry succeeds)={rr['p_success']:.0%}, "
             f"support={rr['support']} events.")
    L.append("")
    L.append("## Ablations — which learned skill contributes what")
    L.append("")
    L.append("| policy | success rate | tool efficiency | avg steps |")
    L.append("|---|---|---|---|")
    for name in ("baseline", "+plans", "+recovery", "learned (full)"):
        a = res["ablations"][name]
        L.append(f"| {name} | {a['success_rate']['mean']:.0%} ± "
                 f"{a['success_rate']['ci95']:.0%} | "
                 f"{a['tool_efficiency']['mean']:.0%} ± "
                 f"{a['tool_efficiency']['ci95']:.0%} | "
                 f"{a['avg_steps']['mean']:.1f} |")
    L.append("")
    L.append("`+plans` isolates planning/tool-selection learned from "
             "trajectories; `+recovery` isolates the mined retry skill; the "
             "full agent composes both.")
    L.append("")
    L.append("## Transfer — leave-one-domain-out")
    L.append("")
    L.append("Memory and recovery are learned WITHOUT the held-out domain, "
             "then evaluated on it. Domain plans should vanish; the "
             "procedural recovery skill should survive.")
    L.append("")
    L.append("| held-out benchmark | baseline | learned (LODO) | lift |")
    L.append("|---|---|---|---|")
    for held, v in res["transfer"].items():
        b, l = v["baseline"]["mean"], v["learned"]["mean"]
        L.append(f"| {BENCHMARKS[held]} | {b:.0%} | {l:.0%} | {l - b:+.0%} |")
    L.append("")
    L.append("## Mined sub-skills (top tool bigrams per domain)")
    L.append("")
    for d, grams in res["skills"].items():
        pretty = ", ".join("→".join(g["gram"]) + f" ({g['count']})"
                           for g in grams[:3])
        L.append(f"- **{BENCHMARKS[d]}**: {pretty}")
    L.append("")
    L.append("*Synthetic trajectories + mock suites (no GPU/network/key in "
             "this sandbox); magnitudes are properties of the mock — the "
             "mechanism, ablation structure, and transfer split are the "
             "result. Real data/benchmarks plug in via sources.py and "
             "benchmark.py.*")
    return "\n".join(L)


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
