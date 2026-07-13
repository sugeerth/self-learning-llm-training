"""Unit tests for the harness 100x levers: kill logic, eval cache, dedupe."""

import math

from harness import EvalCache, dedupe_candidates, eval_key, should_kill


def _cfg(d_model=256, layers=4):
    return {"vocab_size": 50304, "d_model": d_model, "n_layers": layers,
            "n_heads": 4, "n_kv_heads": 2, "d_ff_mult": 8 / 3,
            "max_seq_len": 128, "rope_theta": 10000.0, "dropout": 0.0,
            "tie_embeddings": True}


# ── should_kill ─────────────────────────────────────────────────────────

def test_kill_on_nan_and_inf_immediately():
    assert should_kill([5.0, float("nan")])
    assert should_kill([5.0, float("inf")])


def test_no_kill_during_grace_or_healthy_descent():
    assert not should_kill([9.0, 7.0, 30.0], grace=8)          # within grace
    healthy = [10 - 0.5 * i for i in range(16)]                # descending
    assert not should_kill(healthy, grace=8, factor=2.5)


def test_kill_on_sustained_divergence_but_not_noise():
    diverging = [5.0, 4.0, 3.5, 3.2, 3.0, 3.0, 4.0, 6.0, 9.0, 12.0, 15.0]
    assert should_kill(diverging, grace=8, factor=2.5)
    noisy = [5.0, 4.2, 4.8, 4.0, 4.5, 3.9, 4.4, 4.1, 4.6, 4.0]  # bounded noise
    assert not should_kill(noisy, grace=8, factor=2.5)


# ── eval_key ────────────────────────────────────────────────────────────

def test_eval_key_deterministic_and_ignores_bookkeeping():
    kw = dict(lr=3e-4, batch_size=8, steps_total=16, val_batches=8,
              deterministic_val=True)
    k1 = eval_key(_cfg(), **kw)
    k2 = eval_key({**_cfg(), "_cid": 3, "_eval": {"x": 1}}, **kw)
    assert k1 == k2                                  # _-keys don't affect key


def test_eval_key_sensitive_to_every_input():
    base = dict(lr=3e-4, batch_size=8, steps_total=16, val_batches=8,
                deterministic_val=True)
    k = eval_key(_cfg(), **base)
    assert eval_key(_cfg(d_model=320), **base) != k
    assert eval_key(_cfg(), **{**base, "lr": 1e-3}) != k
    assert eval_key(_cfg(), **{**base, "steps_total": 32}) != k
    assert eval_key(_cfg(), **{**base, "val_batches": 20}) != k
    assert eval_key(_cfg(), **{**base, "deterministic_val": False}) != k
    assert eval_key(_cfg(), torch_seed=7, **base) != k


# ── EvalCache ───────────────────────────────────────────────────────────

def test_cache_roundtrip_and_persistence(tmp_path):
    cache = EvalCache(str(tmp_path / "cache"))
    ev = {"val_ppl": 21.5, "trained_steps": 16, "loss_curve": [3, 2, 1],
          "killed": False}
    cache.put("k1", ev, ckpt_key=None)
    entry = cache.get("k1")
    assert entry["eval"]["val_ppl"] == 21.5
    assert "loss_curve" not in entry["eval"]         # curves not persisted
    # survives a restart
    assert EvalCache(str(tmp_path / "cache")).get("k1")["eval"]["val_ppl"] == 21.5
    assert EvalCache(str(tmp_path / "cache")).get("nope") is None


def test_cache_miss_when_checkpoint_evicted(tmp_path):
    cache = EvalCache(str(tmp_path / "cache"))
    ck = cache.ckpt_path("tk")
    open(ck, "w").write("fake")
    cache.put("k1", {"val_ppl": 20.0, "trained_steps": 8}, ckpt_key="tk")
    assert cache.get("k1") is not None
    import os
    os.remove(ck)                                    # evict the checkpoint
    assert cache.get("k1") is None                   # entry now unusable


def test_cached_inf_ppl_roundtrips(tmp_path):
    cache = EvalCache(str(tmp_path / "cache"))
    cache.put("dead", {"val_ppl": float("inf"), "trained_steps": 9,
                       "killed": True}, ckpt_key=None)
    assert math.isinf(EvalCache(str(tmp_path / "cache"))
                      .get("dead")["eval"]["val_ppl"])


# ── dedupe ──────────────────────────────────────────────────────────────

def test_dedupe_collapses_identical_configs():
    a, b = _cfg(), _cfg(d_model=320)
    reps, alias = dedupe_candidates([a, dict(a), b, dict(a)])
    assert len(reps) == 2
    assert alias == {0: 0, 1: 0, 2: 1, 3: 0}


def test_dedupe_ignores_bookkeeping_keys():
    a = _cfg()
    reps, _ = dedupe_candidates([a, {**a, "_cid": 9, "_steps": 4}])
    assert len(reps) == 1
