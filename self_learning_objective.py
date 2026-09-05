"""self_learning_objective.py — the objective function the system optimizes.

────────────────────────────────────────────────────────────────────────────
WHAT DOES "SELF-LEARNING" MEAN HERE, CONCRETELY?
────────────────────────────────────────────────────────────────────────────
A search method is *self-learning* if it uses what it has already observed to
find good architectures FASTER than blind trial-and-error. We make that
measurable with a single number:

    J  =  speedup vs random search
       =  (steps random needs to reach quality Q) / (steps the learner needs)

where Q is random search's OWN final quality at the shared step budget, measured
per seed (same candidate-RNG luck) on a held-out, deterministic validation set.

    • J > 1  → the learner reaches random's best in fewer steps. It IS learning.
    • J ≈ 1  → no better than random. The "self-learning" is a no-op.
    • J < 1  → the learner is WORSE than random — it is actively doing the wrong
               thing (chasing a misleading surrogate, over-exploiting, etc.).

So "what is the agent doing wrong?" has a precise answer: every extra training
step it spends to reach a quality random already reached for free is *regret*.
The objective is to minimize that regret — equivalently, to maximize J.

────────────────────────────────────────────────────────────────────────────
WHY REGRET-VS-BASELINE (and not raw validation perplexity)?
────────────────────────────────────────────────────────────────────────────
    • Raw perplexity rewards bigger models / more compute, not smarter search.
    • A baseline-relative objective is scale-free and much harder to Goodhart:
      you cannot "win" by inflating the metric, because random is measured on
      the exact same eval, budget, and seeds. The only way to move J is to
      actually spend fewer steps reaching the same quality.

────────────────────────────────────────────────────────────────────────────
ANTI-GOODHART GUARDS  (enforced by `check_comparable`)
────────────────────────────────────────────────────────────────────────────
    1. Held-out eval    — a deterministic fixed-window val set the search never
                          trains on.
    2. Identical eval   — same budget, same val batches, same vocab across arms.
                          (A mismatched softmax denominator silently rescales
                          ppl; arms.py guards this with auto_vocab=False so every
                          arm shares one vocabulary.)
    3. Paired seeds     — each learner seed is judged against the *same-seed*
                          random run, so one seed's RNG luck can't set another's
                          bar.

This module is the single source of truth for the objective. The head-to-head
campaign (arms.py) produces the trajectories; here we turn them into J, decide
pass/fail against a bar, and expose the SAME gate the framework-scout uses to
decide whether adopting a new framework or agent is genuinely an improvement.
Adopt-if-and-only-if-J-improves is what keeps the adaptive system from thrashing
on hype.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict

# The arms whose whole purpose is to beat random by learning from observations.
LEARNING_ARMS = ("prior", "evolve", "agent")
BASELINE_ARM = "random"
# A learner must beat random by at least this multiplicative margin before we
# call the difference "real" rather than seed noise.
DEFAULT_MARGIN = 0.05


# ───────────────────────────── data types ─────────────────────────────

@dataclass
class ObjectiveScore:
    """The objective evaluated on one arms report."""
    value: float                      # J of the best learning arm (higher=better)
    best_arm: str | None              # which learner achieved it
    per_arm: dict[str, float | None]  # speedup_vs_random for every learning arm
    n_seeds: int
    budget_steps: int
    verdict: str                      # "learning" | "no-op" | "regressing" | "invalid"
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GateResult:
    """Pass/fail decision for CI or for framework adoption."""
    passed: bool
    score: float
    threshold: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ───────────────────────────── invariants ─────────────────────────────

def check_comparable(report: dict) -> list[str]:
    """Return a list of HARD anti-Goodhart violations (empty list == the objective
    is trustworthy). These are the conditions under which J actually means what it
    claims; if any fails, the number is not comparable across arms and must not be
    used to accept or reject anything. Soft, confidence-lowering issues (e.g. too
    few seeds) live in `confidence_warnings`, not here."""
    v: list[str] = []
    if not report.get("deterministic_val", False):
        v.append("eval is not deterministic — J is contaminated by eval noise")
    arms = report.get("arms", {})
    if BASELINE_ARM not in arms:
        v.append(f"no '{BASELINE_ARM}' baseline arm — regret target is undefined")
    seed_counts = {a: d.get("seeds", 0) for a, d in arms.items()}
    if seed_counts and len(set(seed_counts.values())) > 1:
        v.append(f"arms ran unequal seed counts {seed_counts} — pairing is broken")
    return v


def confidence_warnings(report: dict) -> list[str]:
    """Soft issues that lower confidence in J without invalidating it — surfaced
    in the rationale so a thin result is never mistaken for a strong one."""
    w: list[str] = []
    arms = report.get("arms", {})
    n = next((d.get("seeds", 0) for d in arms.values()), 0)
    if 0 < n < 5:
        w.append(f"only {n} seed(s) — separates learning from luck weakly "
                 "(recommend >= 5 before trusting a gain/loss verdict)")
    return w


# ───────────────────────────── scoring ─────────────────────────────

def score_report(report: dict) -> ObjectiveScore:
    """Reduce an arms report to the self-learning objective J."""
    violations = check_comparable(report)
    arms = report.get("arms", {})
    budget = int(report.get("budget_steps", 0))
    n_seeds = next((d.get("seeds", 0) for d in arms.values()), 0)

    per_arm: dict[str, float | None] = {}
    for arm in LEARNING_ARMS:
        if arm in arms:
            per_arm[arm] = arms[arm].get("speedup_vs_random")

    present = {a: s for a, s in per_arm.items() if s is not None}
    if violations:
        return ObjectiveScore(
            value=float("nan"), best_arm=None, per_arm=per_arm,
            n_seeds=n_seeds, budget_steps=budget, verdict="invalid",
            rationale="objective not trustworthy: " + "; ".join(violations))
    if not present:
        return ObjectiveScore(
            value=float("nan"), best_arm=None, per_arm=per_arm,
            n_seeds=n_seeds, budget_steps=budget, verdict="invalid",
            rationale="no learning arm reached random's final quality "
                      "(no speedup could be computed)")

    best_arm = max(present, key=present.get)
    J = present[best_arm]
    if J >= 1.0 + DEFAULT_MARGIN:
        verdict, why = "learning", (
            f"best learner '{best_arm}' reaches random's final quality "
            f"{J:.2f}x faster — genuinely sample-efficient")
    elif J <= 1.0 - DEFAULT_MARGIN:
        verdict, why = "regressing", (
            f"best learner '{best_arm}' is {1/J:.2f}x SLOWER than random — "
            "the surrogate/search is actively misleading")
    else:
        verdict, why = "no-op", (
            f"best learner '{best_arm}' is within +-{DEFAULT_MARGIN:.0%} of "
            f"random ({J:.2f}x) — no measurable learning")
    warns = confidence_warnings(report)
    if warns:
        why += "  [low confidence: " + "; ".join(warns) + "]"
    return ObjectiveScore(value=J, best_arm=best_arm, per_arm=per_arm,
                          n_seeds=n_seeds, budget_steps=budget,
                          verdict=verdict, rationale=why)


# ───────────────────────────── the gate ─────────────────────────────

def gate(report: dict, min_speedup: float = 1.0,
         incumbent: float | None = None) -> GateResult:
    """The single decision used by BOTH CI and the framework-scout.

    Passes iff the objective is trustworthy AND the best learner clears the bar:
      • min_speedup — an absolute floor (default 1.0: must at least match random).
      • incumbent   — if given, the candidate must also beat the currently
                      deployed configuration by DEFAULT_MARGIN. This is what a
                      framework/agent swap must satisfy before it is adopted, so
                      a shiny new framework is only swapped in when it measurably
                      reduces regret — never on novelty alone.
    """
    reasons: list[str] = []
    s = score_report(report)
    if s.verdict == "invalid":
        return GateResult(False, s.value, min_speedup, [s.rationale])

    passed = True
    if not (s.value >= min_speedup):
        passed = False
        reasons.append(f"J={s.value:.2f} below floor {min_speedup:.2f}")
    if incumbent is not None and not (s.value >= incumbent + DEFAULT_MARGIN):
        passed = False
        reasons.append(f"J={s.value:.2f} does not beat incumbent "
                       f"{incumbent:.2f} by >= {DEFAULT_MARGIN:.2f}")
    if passed:
        reasons.append(f"{s.verdict}: {s.rationale}")
    return GateResult(passed, s.value, min_speedup, reasons)


# ───────────────────────────── loading ─────────────────────────────

def load_report(path: str | None = None) -> dict:
    """Load an arms report from JSON (defaults to arms.py's REPORT_JSON)."""
    if path is None:
        try:
            from arms import REPORT_JSON
            path = REPORT_JSON
        except Exception:
            path = "arms_report.json"
    with open(path) as f:
        return json.load(f)


def run_fresh_report(seeds: int, budget: int, full_steps: int) -> dict:
    """Run a real (small) arms campaign and return its report — used by the CI
    gate so it measures live behavior instead of a possibly-stale file."""
    from arms import run_arms
    from hyperband import Bracket
    bracket = Bracket(n_candidates=4, halvings=2, initial_steps=max(1, full_steps // 4))
    return run_arms(seeds=seeds, budget=budget, bracket=bracket, full_steps=full_steps)


# ───────────────────────────── CLI ─────────────────────────────

def _cmd_score(args: argparse.Namespace) -> int:
    report = (run_fresh_report(args.seeds, args.budget, args.full_steps)
              if args.run else load_report(args.report))
    s = score_report(report)
    print(json.dumps(s.to_dict(), indent=2))
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    report = (run_fresh_report(args.seeds, args.budget, args.full_steps)
              if args.run else load_report(args.report))
    g = gate(report, min_speedup=args.min_speedup, incumbent=args.incumbent)
    status = "PASS" if g.passed else "FAIL"
    print(f"[objective gate] {status}  J={g.score:.2f}  floor={g.threshold:.2f}")
    for r in g.reasons:
        print(f"  - {r}")
    return 0 if g.passed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Self-learning objective (regret vs random).")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--report", default=None,
                        help="path to arms report JSON (default: arms.REPORT_JSON)")
    common.add_argument("--run", action="store_true",
                        help="run a fresh arms campaign instead of loading a file")
    common.add_argument("--seeds", type=int, default=3)
    common.add_argument("--budget", type=int, default=256)
    common.add_argument("--full-steps", type=int, default=40)

    ps = sub.add_parser("score", parents=[common], help="print the objective J")
    ps.set_defaults(func=_cmd_score)

    pg = sub.add_parser("gate", parents=[common],
                        help="exit non-zero if J is below the bar (for CI)")
    pg.add_argument("--min-speedup", type=float, default=1.0,
                    help="absolute floor on J (default 1.0 = must match random)")
    pg.add_argument("--incumbent", type=float, default=None,
                    help="require J to beat this incumbent J by the margin")
    pg.set_defaults(func=_cmd_gate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
