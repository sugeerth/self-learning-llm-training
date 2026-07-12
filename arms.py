"""Baseline arms — the proof that self-learning works (10X_PLAN Pillar 1).

"Baselines or it didn't happen": every claim about the loop being smart is
tested against dumber arms at an IDENTICAL compute budget (candidate-training
steps) with an IDENTICAL deterministic eval:

  random     candidates sampled uniformly, each trained to full depth
  hyperband  promoted successive halving through the throughput harness
  prior      hyperband + CheapPrior acquisition over oversampled pools,
             prior refit from every full-depth eval across brackets
  agent      prior + the Trainer agent proposes one candidate per bracket
             (needs ANTHROPIC_API_KEY — skipped gracefully without it)

The headline metric is REGRET VS RANDOM: how many training steps each arm
needs to reach the quality random search only reaches by spending its whole
budget. That single number is the "is this actually self-learning?" answer.

CLI:
    python3 arms.py run                    # full: 3 seeds x 256-step budget
    python3 arms.py run --quick            # 2 seeds x 96-step budget (~10 min CPU)

Outputs: arms_report.json + arms_report.html (self-contained SVG regret plot).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from harness import (
    HarnessProfile, load_or_tune, parallel_halving, _train_task, _worker_init,
    _device, RUNS_DIR,
)
from hyperband import Bracket, CheapPrior, random_config

VAL_BATCHES = 8          # deterministic windows — identical for every eval
REPORT_JSON = os.path.join(os.path.dirname(__file__), "arms_report.json")
REPORT_HTML = os.path.join(os.path.dirname(__file__), "arms_report.html")

ARM_COLORS = {"random": "#9aa0a6", "hyperband": "#4285f4",
              "prior": "#34a853", "agent": "#a142f4"}


class Trajectory:
    """Best-so-far quality as a function of cumulative training steps spent."""

    def __init__(self):
        self.points: list[dict] = []   # {steps, ppl, best}
        self.spent = 0
        self.best = math.inf

    def record(self, ev: dict, delta_steps: int) -> None:
        self.spent += delta_steps
        self.best = min(self.best, ev["val_ppl"])
        self.points.append({"steps": self.spent, "ppl": round(ev["val_ppl"], 3),
                            "best": round(self.best, 3)})

    def best_at(self, steps: int) -> float:
        """Best ppl achieved with <= `steps` spent (inf if nothing finished)."""
        best = math.inf
        for p in self.points:
            if p["steps"] <= steps:
                best = p["best"]
        return best

    def steps_to_reach(self, target_ppl: float) -> int | None:
        for p in self.points:
            if p["best"] <= target_ppl:
                return p["steps"]
        return None


# ────────────────────────── arms ──────────────────────────

def run_random(seed: int, budget: int, full_steps: int, profile: HarnessProfile,
               pool: ProcessPoolExecutor) -> Trajectory:
    """Uniform random search: every candidate trained to full depth."""
    rng = random.Random(seed)
    traj = Trajectory()
    n = budget // full_steps
    tasks = [{
        "cfg": random_config(rng), "steps": full_steps, "lr": 3e-4,
        "batch_size": profile.batch_size, "val_batches": VAL_BATCHES,
        "ckpt_in": None, "ckpt_out": None, "want_sample": False,
        "device": _device(), "deterministic_val": True,
    } for _ in range(n)]
    for ev in pool.map(_train_task, tasks):
        traj.record(ev, full_steps)
    return traj


def run_bracketed(seed: int, budget: int, bracket: Bracket,
                  profile: HarnessProfile, pool: ProcessPoolExecutor,
                  use_prior: bool, use_agent: bool) -> Trajectory:
    """Hyperband arm; optionally prior-seeded proposals and agent proposals."""
    rng = random.Random(seed)
    traj = Trajectory()
    prior = CheapPrior() if use_prior else None
    final_depth = bracket.initial_steps * bracket.eta ** bracket.halvings
    history: list[dict] = []

    def on_eval(cfg, ev, delta):
        traj.record(ev, delta)
        if prior is not None and ev["trained_steps"] >= final_depth:
            # only full-depth evals feed the prior — low-depth ppl is a
            # different (noisier) quantity than what we're predicting
            prior.add(cfg, ev["val_ppl"])

    b = 0
    while budget - traj.spent >= _bracket_cost(bracket):
        candidates: list[dict] = []
        if use_agent:
            cand = _agent_proposal(history)
            if cand is not None:
                candidates.append(cand)
        while len(candidates) < bracket.n_candidates:
            if prior is not None and prior.X:
                cands = [random_config(rng) for _ in range(20)]
                cands.sort(key=prior.acquisition)
                candidates.append(cands[0])
            else:
                candidates.append(random_config(rng))
        survivors = parallel_halving(
            candidates, bracket, profile=profile, pool=pool,
            run_dir=os.path.join(RUNS_DIR, f"arms-{seed}-{b}"),
            val_batches_rung=VAL_BATCHES, val_batches_final=VAL_BATCHES,
            want_sample=False, deterministic_val=True, on_eval=on_eval,
        )
        w = survivors[0]
        history.append({"name": f"b{b}-winner", "config": _pub(w),
                        "val_ppl": w["_eval"]["val_ppl"],
                        "params_m": w["_eval"]["params_m"]})
        b += 1
    return traj


def _bracket_cost(bracket: Bracket) -> int:
    """Total training steps one PROMOTED bracket consumes."""
    cost, n, target = 0, bracket.n_candidates, bracket.initial_steps
    prev = 0
    for _ in range(bracket.halvings + 1):
        cost += n * (target - prev)
        prev = target
        target *= bracket.eta
        n = max(1, n // bracket.eta)
    return cost


def _pub(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def _agent_proposal(history: list[dict]) -> dict | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from agents import TrainerAgent
        from dataclasses import asdict
        return asdict(TrainerAgent().propose(history[-10:]))["config"]
    except Exception as e:
        print(f"arms: agent proposal failed ({type(e).__name__}: {e}) — falling back")
        return None


# ────────────────────────── report ──────────────────────────

def _mean_std(xs: list[float]) -> tuple[float, float]:
    finite = [x for x in xs if math.isfinite(x)]
    if not finite:
        return math.inf, 0.0
    m = sum(finite) / len(finite)
    s = (sum((x - m) ** 2 for x in finite) / len(finite)) ** 0.5
    return m, s


def make_report(results: dict[str, list[Trajectory]], budget: int,
                skipped: list[str]) -> dict:
    """results: arm -> one Trajectory per seed."""
    fractions = (0.25, 0.5, 0.75, 1.0)
    # regret target = random search's mean final best; speedups are normalized
    # by the steps RANDOM itself took to first reach that quality (not by the
    # budget — random may hit its final best early, and it must score ~1.0x)
    target = _mean_std([t.best_at(budget) for t in results["random"]])[0]
    rand_steps = [t.steps_to_reach(target) or budget for t in results["random"]]
    baseline_steps = sum(rand_steps) / len(rand_steps)

    arms_out = {}
    for arm, trajs in results.items():
        curve = {}
        for f in fractions:
            m, s = _mean_std([t.best_at(int(budget * f)) for t in trajs])
            curve[f"{int(f * 100)}%"] = {"best_ppl_mean": round(m, 2),
                                         "best_ppl_std": round(s, 2)}
        reached = [t.steps_to_reach(target) for t in trajs]
        reached_steps = [r for r in reached if r is not None]
        arms_out[arm] = {
            "seeds": len(trajs),
            "evals_per_seed": round(sum(len(t.points) for t in trajs) / len(trajs), 1),
            "best_ppl_at_budget": curve,
            "steps_to_random_final": (round(sum(reached_steps) / len(reached_steps))
                                      if len(reached_steps) == len(trajs) else None),
            "trajectories": [t.points for t in trajs],
        }
        s2r = arms_out[arm]["steps_to_random_final"]
        arms_out[arm]["speedup_vs_random"] = (round(baseline_steps / s2r, 2)
                                              if s2r else None)
    return {"budget_steps": budget, "regret_target_ppl": round(target, 2),
            "random_steps_to_target": round(baseline_steps),
            "val_batches": VAL_BATCHES, "deterministic_val": True,
            "skipped_arms": skipped, "arms": arms_out}


def print_report(report: dict) -> None:
    print(f"\narms: budget={report['budget_steps']} steps/arm/seed, "
          f"regret target (random's final best) = ppl {report['regret_target_ppl']}")
    hdr = f"{'arm':<10} {'evals':>6} {'best@50%':>10} {'best@100%':>10} {'steps→target':>13} {'speedup':>8}"
    print(hdr + "\n" + "-" * len(hdr))
    for arm, a in report["arms"].items():
        s2r = a["steps_to_random_final"]
        print(f"{arm:<10} {a['evals_per_seed']:>6} "
              f"{a['best_ppl_at_budget']['50%']['best_ppl_mean']:>10} "
              f"{a['best_ppl_at_budget']['100%']['best_ppl_mean']:>10} "
              f"{s2r if s2r else 'not reached':>13} "
              f"{str(a['speedup_vs_random']) + 'x' if a['speedup_vs_random'] else '—':>8}")
    if report["skipped_arms"]:
        print(f"arms: skipped (no ANTHROPIC_API_KEY): {', '.join(report['skipped_arms'])}")


def write_html(report: dict) -> None:
    """Self-contained regret plot: best-ppl-so-far vs steps, one line per
    arm-seed, no external deps."""
    W, H, PAD = 720, 420, 50
    budget = report["budget_steps"]
    pts = [p for a in report["arms"].values() for t in a["trajectories"] for p in t]
    if not pts:
        return
    lo = min(p["best"] for p in pts)
    hi = max(p["best"] for p in pts)
    span = max(hi - lo, 1e-9)

    def X(s):
        return PAD + (W - 2 * PAD) * s / budget

    def Y(v):  # standard axis: low ppl (good) at the bottom
        return H - PAD - (H - 2 * PAD) * (v - lo) / span

    lines, legend = [], []
    for i, (arm, a) in enumerate(report["arms"].items()):
        color = ARM_COLORS.get(arm, "#666")
        for t in a["trajectories"]:
            if not t:
                continue
            d = f"M {X(t[0]['steps']):.1f} {Y(t[0]['best']):.1f} " + " ".join(
                f"L {X(p['steps']):.1f} {Y(p['best']):.1f}" for p in t[1:])
            lines.append(f'<path d="{d}" fill="none" stroke="{color}" '
                         f'stroke-width="2" opacity="0.75"/>')
        legend.append(f'<text x="{PAD + 130 * i}" y="{PAD - 18}" fill="{color}" '
                      f'font-weight="600">{arm}</text>')
    tgt = report["regret_target_ppl"]
    target_line = (f'<line x1="{PAD}" y1="{Y(tgt):.1f}" x2="{W - PAD}" y2="{Y(tgt):.1f}" '
                   f'stroke="#d93025" stroke-dasharray="6 4"/>'
                   f'<text x="{W - PAD - 4}" y="{Y(tgt) - 6:.1f}" fill="#d93025" '
                   f'text-anchor="end" font-size="12">random final ({tgt})</text>')
    html = f"""<!-- generated by arms.py — regret vs random -->
<h2 style="font-family:system-ui">Baseline arms — best val ppl vs training steps spent</h2>
<p style="font-family:system-ui;max-width:{W}px">Every arm gets {budget} training
steps and the same deterministic eval. Lower is better; the dashed line is
random search's final quality — the steps an arm needs to cross it is its
speedup over random. Thin lines are individual seeds.</p>
<svg viewBox="0 0 {W} {H}" width="{W}" style="font-family:system-ui;background:#fff">
  <rect x="{PAD}" y="{PAD}" width="{W - 2 * PAD}" height="{H - 2 * PAD}"
        fill="none" stroke="#ddd"/>
  {''.join(legend)}
  {target_line}
  {''.join(lines)}
  <text x="{W / 2}" y="{H - 12}" text-anchor="middle" font-size="13">training steps spent</text>
  <text x="14" y="{H / 2}" font-size="13" transform="rotate(-90 14 {H / 2})"
        text-anchor="middle">best val ppl (deterministic)</text>
  <text x="{PAD - 6}" y="{Y(lo) + 4:.1f}" text-anchor="end" font-size="12">{lo:.0f}</text>
  <text x="{PAD - 6}" y="{Y(hi) + 4:.1f}" text-anchor="end" font-size="12">{hi:.0f}</text>
  <text x="{PAD}" y="{H - PAD + 16}" font-size="12">0</text>
  <text x="{W - PAD}" y="{H - PAD + 16}" text-anchor="end" font-size="12">{budget}</text>
</svg>
"""
    with open(REPORT_HTML, "w") as f:
        f.write(html)


# ────────────────────────── driver ──────────────────────────

def run_arms(seeds: int, budget: int, bracket: Bracket, full_steps: int) -> dict:
    profile = load_or_tune(quick=True)
    from data import prepare
    prepare()  # once, before pool spawn

    arm_fns = {
        "random": lambda s, pool: run_random(s, budget, full_steps, profile, pool),
        "hyperband": lambda s, pool: run_bracketed(s, budget, bracket, profile, pool,
                                                   use_prior=False, use_agent=False),
        "prior": lambda s, pool: run_bracketed(s, budget, bracket, profile, pool,
                                               use_prior=True, use_agent=False),
    }
    skipped = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        arm_fns["agent"] = lambda s, pool: run_bracketed(s, budget, bracket, profile,
                                                         pool, use_prior=True,
                                                         use_agent=True)
    else:
        skipped.append("agent")

    results: dict[str, list[Trajectory]] = {a: [] for a in arm_fns}
    with ProcessPoolExecutor(max_workers=profile.workers,
                             mp_context=get_context("spawn"),
                             initializer=_worker_init,
                             initargs=(profile.threads_per_worker,)) as pool:
        for arm, fn in arm_fns.items():
            for seed in range(seeds):
                t0 = time.time()
                traj = fn(seed, pool)
                results[arm].append(traj)
                print(f"arms: {arm:<10} seed={seed} best={traj.best:8.2f} "
                      f"spent={traj.spent:>4} steps  evals={len(traj.points):>3} "
                      f"({time.time() - t0:.0f}s)")

    report = make_report(results, budget, skipped)
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    write_html(report)
    print_report(report)
    print(f"arms: wrote {REPORT_JSON} and {REPORT_HTML}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run all arms at an equal step budget")
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--budget", type=int, default=256, help="training steps per arm per seed")
    r.add_argument("--quick", action="store_true", help="2 seeds x 96 steps (~10 min CPU)")
    args = ap.parse_args()

    if args.cmd == "run":
        seeds, budget = args.seeds, args.budget
        if args.quick:
            seeds, budget = 2, 96
        bracket = Bracket(n_candidates=4, halvings=2, initial_steps=max(2, budget // 24))
        full_steps = bracket.initial_steps * bracket.eta ** bracket.halvings
        run_arms(seeds=seeds, budget=budget, bracket=bracket, full_steps=full_steps)


if __name__ == "__main__":
    main()
