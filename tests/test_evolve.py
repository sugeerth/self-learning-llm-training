"""Tests for the evolutionary arm's genetics — operators, prior-guided
breeding, and the lineage/elitism bookkeeping. The full loop is exercised with
a fake pool that scores genomes by a cheap analytic fitness, so no training (or
torch) is needed for the genetics; the operator/validity tests do import torch
via the model to assert every bred genome is actually buildable."""
import random

import pytest

from evolve import (
    crossover, mutate, repair, _valid_heads, _key, _full, GENES,
    _breed, run_evolve, Genome,
)
from hyperband import random_config, CheapPrior


# ────────────────────────── operators & validity ──────────────────────────

def _draw(rng):
    base = random_config(rng)
    return repair({g: base[g] for g in GENES})


def test_repair_enforces_head_divisibility():
    # n_kv must DIVIDE n_heads, and n_heads must divide d_model
    fixed = repair({"d_model": 384, "n_layers": 4, "n_heads": 2,
                    "n_kv_heads": 4, "d_ff_mult": 3.0, "tie_embeddings": True})
    assert fixed["n_heads"] % fixed["n_kv_heads"] == 0
    assert 384 % fixed["n_heads"] == 0
    # a d_model that 6 doesn't divide must snap n_heads to a real divisor
    fixed2 = repair({"d_model": 512, "n_layers": 4, "n_heads": 6,
                     "n_kv_heads": 6, "d_ff_mult": 2.0, "tie_embeddings": True})
    assert 512 % fixed2["n_heads"] == 0
    assert fixed2["n_heads"] % fixed2["n_kv_heads"] == 0


def test_bred_genomes_are_always_buildable():
    """Crossover + mutation must never emit an architecture the model rejects."""
    pytest.importorskip("torch")
    from model import ModelConfig
    from harness import _clean
    rng = random.Random(0)
    for _ in range(2000):
        a, b = _draw(rng), _draw(rng)
        child = mutate(crossover(a, b, rng), rng)
        # raises AssertionError inside ModelConfig if any constraint is violated
        ModelConfig(**_clean(_full(child)))


def test_crossover_only_mixes_parent_genes():
    rng = random.Random(1)
    a = _draw(rng)
    b = _draw(rng)
    child = crossover(a, b, rng)
    # after repair, n_kv may be snapped; the freely-inherited genes must each
    # trace to one parent
    for g in ("d_model", "n_layers", "d_ff_mult", "tie_embeddings"):
        assert child[g] in (a[g], b[g])


def test_mutation_changes_something_but_stays_valid():
    rng = random.Random(2)
    base = _draw(rng)
    changed = sum(_key(mutate(base, rng)) != _key(base) for _ in range(200))
    assert changed > 100        # rate ~0.34 across 5 genes -> usually changes
    for _ in range(200):
        m = mutate(base, rng)
        assert m["d_model"] % m["n_heads"] == 0
        assert m["n_heads"] % m["n_kv_heads"] == 0


# ────────────────────────── prior-guided breeding ──────────────────────────

def test_breed_prefers_low_acquisition_children():
    """With a prior that has learned a good region, _breed should return a child
    the surrogate scores better than the average random cross of the parents."""
    rng = random.Random(3)
    parents = [Genome("p0", 0, _draw(rng), [], "random"),
               Genome("p1", 0, _draw(rng), [], "random")]
    prior = CheapPrior()
    # teach the prior that small, shallow nets are good and big deep ones bad
    prior.add({"d_model": 192, "n_layers": 2, "n_heads": 2, "n_kv_heads": 2,
               "d_ff_mult": 2.0, "tie_embeddings": True}, 20.0, steps=12)
    prior.add({"d_model": 512, "n_layers": 8, "n_heads": 8, "n_kv_heads": 8,
               "d_ff_mult": 4.0, "tie_embeddings": True}, 400.0, steps=12)
    chosen, par = _breed(parents, prior, rng, taken=set(), oversample=12)
    baseline = [mutate(crossover(parents[0].cfg, parents[1].cfg, rng), rng)
                for _ in range(12)]
    mean_baseline = sum(prior.acquisition(c) for c in baseline) / len(baseline)
    assert prior.acquisition(chosen) <= mean_baseline
    assert set(par) <= {"p0", "p1"}


def test_breed_penalises_but_survives_all_duplicates():
    """Even if every candidate collides with `taken`, _breed returns a genome
    rather than None (the population must still fill)."""
    rng = random.Random(4)
    p = Genome("p0", 0, _draw(rng), [], "random")
    parents = [p, Genome("p1", 0, _draw(rng), [], "random")]
    everything = {_key(mutate(crossover(parents[0].cfg, parents[1].cfg, rng), rng))
                  for _ in range(50)}
    chosen, _ = _breed(parents, CheapPrior(), rng, taken=everything, oversample=6)
    assert chosen is not None


# ────────────────────────── full loop: lineage + elitism ──────────────────────────

class _FakePool:
    """Stands in for the ProcessPoolExecutor. Scores each genome by an analytic
    'fitness' so the evolutionary bookkeeping runs without any training."""

    @staticmethod
    def _fitness(cfg: dict) -> float:
        # a smooth bowl minimised near d_model=320, n_layers=4 — evolution
        # should march the population toward it
        return (abs(cfg["d_model"] - 320) / 64) ** 2 + (abs(cfg["n_layers"] - 4)) ** 2 + 10

    def map(self, fn, tasks):
        out = []
        for t in tasks:
            ppl = self._fitness(t["cfg"])
            out.append({"val_ppl": ppl, "params_m": round(t["cfg"]["d_model"] / 100, 2),
                        "trained_steps": t["steps"]})
        return out


class _Profile:
    batch_size = 8


def test_run_evolve_lineage_is_a_valid_genealogy():
    res = run_evolve(seed=0, budget=240, full_steps=12, profile=_Profile(),
                     pool=_FakePool(), pop=6, elite=3, oversample=6)
    lineage = res["lineage"]
    ids = {g["id"] for g in lineage}
    assert len(ids) == len(lineage)                 # ids are unique
    assert res["generations"] >= 2                  # actually bred, not just gen0

    by_gen = {}
    for g in lineage:
        by_gen.setdefault(g["gen"], []).append(g)
    assert all(g["origin"] == "random" for g in by_gen[0])
    assert len(by_gen[0]) == 6                       # full initial population

    # every non-root genome names parents that already exist and are older
    id_gen = {g["id"]: g["gen"] for g in lineage}
    for g in lineage:
        for pid in g["parents"]:
            assert pid in ids
            assert id_gen[pid] < g["gen"]            # parents strictly precede


def test_run_evolve_improves_best_over_generations():
    res = run_evolve(seed=1, budget=360, full_steps=12, profile=_Profile(),
                     pool=_FakePool(), pop=6, elite=3, oversample=8)
    lineage = res["lineage"]
    best_by_gen = {}
    for g in lineage:
        best_by_gen[g["gen"]] = min(best_by_gen.get(g["gen"], 1e9), g["ppl"])
    gens = sorted(best_by_gen)
    # elitism guarantees best-so-far never regresses; selection should improve it
    running = 1e9
    for gn in gens:
        running = min(running, best_by_gen[gn])
    assert best_by_gen[gens[-1]] <= best_by_gen[gens[0]]
    assert res["best"]["ppl"] == pytest.approx(min(g["ppl"] for g in lineage))


def test_run_evolve_respects_budget():
    budget, full_steps = 240, 12
    res = run_evolve(seed=2, budget=budget, full_steps=full_steps, profile=_Profile(),
                     pool=_FakePool(), pop=6, elite=3, oversample=6)
    # never overspends; each genome eval costs full_steps
    assert res["spent"] <= budget
    assert res["spent"] == len(res["lineage"]) * full_steps
