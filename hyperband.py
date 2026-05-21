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
    """
    def __init__(self, length_scale: float = 0.5):
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        self.l = length_scale

    @staticmethod
    def featurize(cfg: dict) -> np.ndarray:
        return np.array([
            math.log(cfg.get("n_layers", 6)),
            math.log(cfg.get("d_model", 384)),
            math.log(max(cfg.get("n_heads", 6), 1)),
            math.log(max(cfg.get("n_kv_heads", 2), 1)),
            cfg.get("d_ff_mult", 8/3),
            float(cfg.get("tie_embeddings", True)),
        ], dtype=float)

    def add(self, cfg: dict, val_ppl: float) -> None:
        self.X.append(self.featurize(cfg))
        self.y.append(math.log(val_ppl))  # log-space — ppl is heavy-tailed

    def predict(self, cfg: dict) -> tuple[float, float]:
        """Return (mean log-ppl, uncertainty)."""
        if not self.X:
            return (5.0, 1.0)
        x = self.featurize(cfg)
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
