"""Hyperband + Bayesian-prior adaptive scheduler for LLM sweep.

Self-learning loop:
  1. Load priors (experiments.json) → fit a simple GP over (params, n_layers, d_model) → val_ppl
  2. Use the GP to seed proposals; trainer agent perturbs from there
  3. Successive halving: train all candidates short → kill bottom half → train survivors longer
  4. After each round, refit the GP — the model literally learns which arch to try next

This is the "self-learning" core. No external deps (numpy only).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Bracket:
    """One Hyperband bracket: start with N candidates, halve s times, total budget B."""
    n_candidates: int
    halvings: int
    initial_steps: int
    eta: int = 2  # halving factor


def standard_brackets(max_steps: int = 200, eta: int = 2) -> list[Bracket]:
    """Generate Hyperband brackets up to max_steps. Each bracket trades breadth↔depth."""
    s_max = int(math.floor(math.log(max_steps) / math.log(eta)))
    brackets = []
    for s in range(s_max, -1, -1):
        n = int(math.ceil((s_max + 1) / (s + 1) * eta**s))
        r = max(1, max_steps // (eta**s))
        brackets.append(Bracket(n_candidates=n, halvings=s, initial_steps=r, eta=eta))
    return brackets


# ── tiny GP-ish prior over (n_layers, d_model, n_heads) → val_ppl ─────────

class CheapPrior:
    """Not a real GP — RBF-kernel weighted average of prior runs.
    Self-improves: every new (config, ppl) pair is added to the support set.

    Depth-aware (v2): evals carry the training depth (steps) they were
    measured at as an extra feature, so EVERY rung eval can feed the prior —
    a shallow eval of the same arch contributes, just with less weight than a
    full-depth one. (v1 could only learn from ~2 full-depth evals per bracket
    and starved; feeding it mixed depths without the feature would conflate
    "bad arch" with "barely trained".)
    """
    def __init__(self, length_scale: float = 0.5):
        self.support: list[tuple[dict, float, float]] = []   # (cfg, ppl, steps)
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        self.l = length_scale
        self._max_steps = 1.0    # deepest eval seen; queries default to it

    @staticmethod
    def featurize(cfg: dict, steps: float | None = None) -> np.ndarray:
        return np.array([
            math.log(cfg.get("n_layers", 6)),
            math.log(cfg.get("d_model", 384)),
            math.log(max(cfg.get("n_heads", 6), 1)),
            math.log(max(cfg.get("n_kv_heads", 2), 1)),
            cfg.get("d_ff_mult", 8/3),
            float(cfg.get("tie_embeddings", True)),
            math.log(max(steps or 1.0, 1.0)) / 2,   # /2: depth informs, not dominates
        ], dtype=float)

    def add(self, cfg: dict, val_ppl: float, steps: float | None = None) -> None:
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        steps = float(steps) if steps else self._max_steps
        self._max_steps = max(self._max_steps, steps)
        self.support.append((cfg, float(val_ppl), steps))
        self.X.append(self.featurize(cfg, steps))
        self.y.append(math.log(val_ppl))  # log-space — ppl is heavy-tailed

    def predict(self, cfg: dict, steps: float | None = None) -> tuple[float, float]:
        """Return (mean log-ppl, uncertainty) at `steps` depth (default: the
        deepest depth seen — 'how good would this arch be fully trained')."""
        if not self.X:
            return (5.0, 1.0)
        x = self.featurize(cfg, steps or self._max_steps)
        Xa = np.stack(self.X)
        d2 = ((Xa - x) ** 2).sum(axis=1)
        w = np.exp(-d2 / (2 * self.l ** 2))
        if w.sum() < 1e-6:
            return (float(np.mean(self.y)), 1.0)
        mean = float((w * np.array(self.y)).sum() / w.sum())
        unc = float(1.0 / (1.0 + w.sum()))  # higher when no nearby support
        return mean, unc

    def acquisition(self, cfg: dict, kappa: float = 1.5) -> float:
        """Lower-confidence-bound (we minimize ppl): mean - kappa * unc.
        Smaller = more attractive."""
        mean, unc = self.predict(cfg)
        return mean - kappa * unc

    # ── persistence: the prior starves inside one run; let it compound ──
    # Raw (cfg, ppl, steps) triples are stored, not feature vectors, so the
    # featurization can evolve without invalidating accumulated knowledge.

    def save(self, path: str = "prior_store.json") -> None:
        with open(path, "w") as f:
            json.dump({"version": 2, "length_scale": self.l,
                       "support": [{"cfg": c, "ppl": p, "steps": s}
                                   for c, p, s in self.support]}, f)

    @classmethod
    def load(cls, path: str = "prior_store.json") -> "CheapPrior":
        """Missing/corrupt/old-format file -> empty prior (never raises)."""
        p = cls()
        try:
            with open(path) as f:
                d = json.load(f)
            p.l = d.get("length_scale", p.l)
            for row in d.get("support", []):
                p.add(row["cfg"], row["ppl"], row.get("steps"))
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            pass
        return p

    def extend(self, other: "CheapPrior") -> None:
        for cfg, ppl, steps in other.support:
            self.add(cfg, ppl, steps)


# ── successive-halving runner ──────────────────────────────────────────

def successive_halving(
    candidates: list[dict],
    train_partial: Callable[[dict, int], dict],
    bracket: Bracket,
) -> list[dict]:
    """Run a single Hyperband bracket. train_partial(cfg, steps) -> {val_ppl, ...}.
    Returns surviving candidates with their final eval dict attached as cfg['_eval']."""
    survivors = list(candidates)
    steps = bracket.initial_steps
    for halving in range(bracket.halvings + 1):
        for cfg in survivors:
            ev = train_partial(cfg, steps)
            cfg["_eval"] = ev
            cfg["_steps"] = cfg.get("_steps", 0) + steps
        survivors.sort(key=lambda c: c["_eval"]["val_ppl"])
        if halving < bracket.halvings:
            keep = max(1, len(survivors) // bracket.eta)
            survivors = survivors[:keep]
            steps *= bracket.eta
    return survivors


# ── seeded proposal generator (used by trainer agent as fallback) ──────

def random_config(rng: random.Random) -> dict:
    d_model = rng.choice([192, 256, 320, 384, 448, 512])
    n_heads = rng.choice([h for h in [2, 4, 6, 8] if d_model % h == 0])
    n_kv = rng.choice([1, 2, n_heads])
    return {
        "vocab_size": 50304,
        "d_model": d_model,
        "n_layers": rng.choice([2, 4, 6, 8]),
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "d_ff_mult": rng.choice([2.0, 8 / 3, 3.0, 4.0]),
        "max_seq_len": 128,
        "rope_theta": 10000.0,
        "dropout": 0.0,
        "tie_embeddings": True,
    }


def load_prior_from_experiments(path: str = "experiments.json") -> CheapPrior:
    p = CheapPrior()
    try:
        with open(path) as f:
            d = json.load(f)
        for v in d.get("variants", []):
            p.add(v["config"], v["val_ppl"])
    except FileNotFoundError:
        pass
    return p
