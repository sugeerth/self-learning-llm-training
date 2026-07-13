"""Torch-free tests for the search core (hyperband.py)."""
import math
import random

import pytest

from hyperband import Bracket, CheapPrior, random_config, standard_brackets, successive_halving


def test_standard_brackets_trade_breadth_for_depth():
    brackets = standard_brackets(max_steps=200, eta=2)
    assert brackets, "must generate at least one bracket"
    # earlier brackets: many candidates, few initial steps; later: the reverse
    assert brackets[0].n_candidates >= brackets[-1].n_candidates
    assert brackets[0].initial_steps <= brackets[-1].initial_steps
    for b in brackets:
        assert b.n_candidates >= 1 and b.initial_steps >= 1


def test_random_config_is_always_valid():
    rng = random.Random(0)
    for _ in range(200):
        cfg = random_config(rng)
        assert cfg["d_model"] % cfg["n_heads"] == 0
        assert cfg["n_kv_heads"] <= cfg["n_heads"]
        assert cfg["n_layers"] >= 1


def test_cheap_prior_learns_and_prefers_known_good_region():
    prior = CheapPrior()
    good = {"n_layers": 8, "d_model": 512, "n_heads": 8, "n_kv_heads": 2,
            "d_ff_mult": 8 / 3, "tie_embeddings": True}
    bad = {"n_layers": 2, "d_model": 192, "n_heads": 2, "n_kv_heads": 1,
           "d_ff_mult": 2.0, "tie_embeddings": True}
    for _ in range(3):
        prior.add(good, 20.0)
        prior.add(bad, 200.0)
    mean_good, unc_good = prior.predict(good)
    mean_bad, unc_bad = prior.predict(bad)
    assert mean_good < mean_bad          # log-ppl ordering preserved
    assert unc_good < 1.0                # support nearby -> some confidence
    # acquisition (lower = more attractive) must prefer the good region
    assert prior.acquisition(good) < prior.acquisition(bad)


def test_cheap_prior_empty_is_uninformative():
    prior = CheapPrior()
    mean, unc = prior.predict({"n_layers": 4, "d_model": 256, "n_heads": 4})
    assert unc == pytest.approx(1.0)
    assert math.isfinite(mean)


def test_successive_halving_kills_bottom_half_and_attaches_evals():
    calls = []

    def fake_train(cfg, steps):
        calls.append((cfg["id"], steps))
        return {"val_ppl": cfg["quality"] + steps * 0}  # quality decides survival

    candidates = [{"id": i, "quality": float(i)} for i in range(4)]
    bracket = Bracket(n_candidates=4, halvings=2, initial_steps=5, eta=2)
    survivors = successive_halving(candidates, fake_train, bracket)

    assert len(survivors) == 1
    assert survivors[0]["id"] == 0                     # best quality wins
    assert survivors[0]["_eval"]["val_ppl"] == 0.0
    # rung structure: 4 evals @5, 2 @10, 1 @20
    steps_seen = sorted(s for _, s in calls)
    assert steps_seen == [5, 5, 5, 5, 10, 10, 20]
