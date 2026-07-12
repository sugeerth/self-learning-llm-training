"""Tests for the flywheel's offline filters and mixing (numpy only)."""
import numpy as np
import pytest

from flywheel import (
    dedup, distinct_ratio, heuristic_filter, mix_corpus, repetition_fraction,
)
from hyperband import CheapPrior


def test_repetition_gate_kills_degenerate_loops():
    loop = [1, 2, 3] * 30                      # classic degenerate generation
    diverse = list(range(90))
    assert repetition_fraction(loop) > 0.9
    assert repetition_fraction(diverse) == 0.0


def test_distinct_ratio_flags_low_diversity():
    monotone = [7] * 60
    diverse = list(range(60))
    assert distinct_ratio(monotone) < 0.1
    assert distinct_ratio(diverse) == 1.0


def test_dedup_drops_ngram_overlaps_keeps_first():
    a = list(range(0, 20))
    b = list(range(100, 120))
    a_copyish = list(range(5, 25))             # shares 8-grams with a
    kept = dedup([a, a_copyish, b])
    assert kept == [a, b]


def test_heuristic_filter_chain_and_stats():
    good = [list(range(i, i + 40)) for i in range(0, 400, 100)]
    bad = [[1, 2] * 20, [9] * 40]
    kept, stats = heuristic_filter(good + bad)
    assert kept == good
    assert stats["generated"] == 6
    assert stats["after_dedup"] == 4
    assert stats["kept_rate"] == pytest.approx(4 / 6, abs=0.01)


def test_mix_corpus_caps_at_ratio_and_preserves_real_prefix(tmp_path):
    real = np.arange(1000, dtype=np.uint16)
    samples = [list(range(2000, 2080))] * 10   # plenty of synthetic supply
    out = str(tmp_path / "mixed.bin")
    info = mix_corpus(real, samples, ratio=0.1, out_path=out)

    mixed = np.fromfile(out, dtype=np.uint16)
    assert (mixed[:1000] == real).all()        # real data intact, first
    assert info["synth_tokens"] <= 100 + 1     # cap + separator
    assert info["ratio_actual"] <= 0.11


def test_cheap_prior_persistence_roundtrip(tmp_path):
    p = CheapPrior()
    cfg = {"n_layers": 4, "d_model": 256, "n_heads": 4, "n_kv_heads": 2,
           "d_ff_mult": 8 / 3, "tie_embeddings": True}
    p.add(cfg, 42.0)
    path = str(tmp_path / "prior.json")
    p.save(path)

    q = CheapPrior.load(path)
    assert len(q.X) == 1
    assert q.predict(cfg) == pytest.approx(p.predict(cfg))
    # missing file -> empty prior, never raises
    assert CheapPrior.load(str(tmp_path / "nope.json")).X == []
