"""Agent trajectory schema + IO + a synthetic generator.

A Trajectory is one agent's attempt at one task: an ordered list of Steps
(thought -> tool action -> observation), an outcome, and cost/runtime/tool
telemetry. This is the unit we collect from public runs, score, retrieve, and
learn from.

Tasks come in DOMAINS (gaia/mle/swe) and, within a domain, FAMILIES — variants
whose correct tool sequence differs and is signalled by a keyword in the goal
(e.g. a GAIA "convert" task needs the calculator; a "lookup" task doesn't). A
memoryless agent can't tell families apart; an agent that has learned from past
trajectories can. That gap is exactly the skill this project studies.

Real public trajectories (SWE-bench runs, GAIA logs, ML-agent traces) plug in
via `sources.py`; the synthetic generator here lets the whole pipeline run and
be tested offline — no network, no GPU, no API key.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field

DOMAIN_TOOLS = {
    "gaia": ["search", "browse", "read", "extract", "calculator", "answer"],
    "mle":  ["load_data", "eda", "train", "tune", "evaluate", "submit"],
    "swe":  ["locate", "read_code", "edit", "run_tests", "debug", "commit"],
}

# (domain, family) -> optimal tool sequence. Family is inferred from the goal.
FAMILIES = {
    ("gaia", "lookup"):  ["search", "read", "extract", "answer"],
    ("gaia", "convert"): ["search", "read", "calculator", "answer"],
    ("mle", "baseline"): ["load_data", "eda", "train", "evaluate", "submit"],
    ("mle", "tuned"):    ["load_data", "eda", "train", "tune", "evaluate", "submit"],
    ("swe", "fix"):      ["locate", "read_code", "edit", "run_tests", "commit"],
    ("swe", "debug"):    ["locate", "read_code", "debug", "edit", "run_tests", "commit"],
}

# keyword in the goal -> family (how an agent could infer the plan from the task)
_FAMILY_CUE = {
    "convert": ("gaia", "convert"), "mass": ("gaia", "convert"),
    "tune": ("mle", "tuned"), "optimize": ("mle", "tuned"),
    "flaky": ("swe", "debug"), "intermittent": ("swe", "debug"),
}
_DEFAULT_FAMILY = {"gaia": "lookup", "mle": "baseline", "swe": "fix"}


def family_of(domain: str, goal: str) -> str:
    g = goal.lower()
    for cue, (dom, fam) in _FAMILY_CUE.items():
        if dom == domain and cue in g:
            return fam
    return _DEFAULT_FAMILY[domain]


def canonical_for(domain: str, goal: str) -> list[str]:
    return FAMILIES[(domain, family_of(domain, goal))]


@dataclass
class Step:
    thought: str
    tool: str
    observation: str
    ok: bool = True


@dataclass
class Trajectory:
    task_id: str
    domain: str
    goal: str
    steps: list[Step] = field(default_factory=list)
    success: bool = False
    cost_usd: float = 0.0
    runtime_s: float = 0.0
    source: str = "synthetic"

    @property
    def tools(self) -> list[str]:
        return [s.tool for s in self.steps]

    @property
    def recovered(self) -> bool:
        return self.success and any(not s.ok for s in self.steps)

    def text(self) -> str:
        return f"{self.domain} {self.goal} :: {' '.join(self.tools)}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls(**{**d, "steps": [Step(**s) for s in d.get("steps", [])]})


def save_jsonl(trajs: list[Trajectory], path: str) -> None:
    with open(path, "w") as f:
        for t in trajs:
            f.write(json.dumps(t.to_dict()) + "\n")


def load_jsonl(path: str) -> list[Trajectory]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(Trajectory.from_dict(json.loads(line)))
    return out


# ────────────────────────── synthetic generator ──────────────────────────

_GOALS = {
    ("gaia", "lookup"):  ["Find the population of a capital",
                          "Find a book's author", "Find a historical date"],
    ("gaia", "convert"): ["Convert and compute a currency amount",
                          "Compute a chemical mass", "Convert units and answer"],
    ("mle", "baseline"): ["Build a tabular classifier", "Fit a regression baseline",
                          "Train a churn predictor"],
    ("mle", "tuned"):    ["Train and tune a classifier", "Optimize model hyperparams",
                          "Tune a forecaster"],
    ("swe", "fix"):      ["Fix a failing unit test", "Fix an off-by-one error",
                          "Fix a config parsing bug"],
    ("swe", "debug"):    ["Debug a flaky test", "Trace an intermittent crash",
                          "Debug an intermittent regression"],
}


def _steps_from_plan(plan: list[str]) -> list[Step]:
    return [Step(thought=f"use {t}", tool=t, observation=f"{t} done") for t in plan]


def synth_trajectory(domain: str, family: str, i: int,
                     rng: random.Random) -> Trajectory:
    goal = f"{rng.choice(_GOALS[(domain, family)])} (case {i})"
    plan = list(FAMILIES[(domain, family)])
    roll = rng.random()
    success = roll > 0.30
    if success and roll < 0.55:                # recovery: a wrong step, then success
        wrong = rng.choice([t for t in DOMAIN_TOOLS[domain] if t != plan[0]])
        steps = [Step("try", wrong, "dead end", ok=False)] + _steps_from_plan(plan)
    elif success:
        steps = _steps_from_plan(plan)
    else:                                      # failure: stop short + stray action
        steps = _steps_from_plan(plan[:-1]) + [
            Step("guess", rng.choice(DOMAIN_TOOLS[domain]), "no progress", ok=False)]
    n = max(len(steps), 1)
    return Trajectory(task_id=f"{domain}-{family}-{i}", domain=domain, goal=goal,
                      steps=steps, success=success,
                      cost_usd=round(0.002 * n, 4), runtime_s=round(1.5 * n, 1))


def synth_dataset(n_per_family: int = 20, seed: int = 0) -> list[Trajectory]:
    rng = random.Random(seed)
    out = []
    for (domain, family) in FAMILIES:
        for i in range(n_per_family):
            out.append(synth_trajectory(domain, family, i, rng))
    return out
