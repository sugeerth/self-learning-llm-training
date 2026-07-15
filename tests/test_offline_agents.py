"""The multi-agent loop with no API key: local agent-brain decisions.

Torch-free — these exercise the offline reasoning, not the training.
"""

import math

from offline_agents import (offline_audit, offline_meta, offline_propose,
                            offline_score)


def _cfg(d_model=256):
    heads = 4
    return {"vocab_size": 128, "d_model": d_model, "n_layers": 4,
            "n_heads": heads, "n_kv_heads": 2, "d_ff_mult": 8 / 3,
            "max_seq_len": 128, "rope_theta": 10000.0, "dropout": 0.0,
            "tie_embeddings": True}


# ── Trainer ──────────────────────────────────────────────────────────────

def test_propose_cold_start_is_valid_and_deterministic():
    a = offline_propose([])
    b = offline_propose([])
    assert a == b                               # deterministic per state
    assert a["config"]["d_model"] % a["config"]["n_heads"] == 0   # trainable
    assert a["config"]["n_kv_heads"] in (1, 2, a["config"]["n_heads"])
    assert a["lr"] > 0 and a["name"]


def test_propose_uses_history_and_avoids_duplicate_names():
    hist = [{"name": "a", "config": _cfg(256), "val_ppl": 30.0, "params_m": 6.0},
            {"name": "b", "config": _cfg(384), "val_ppl": 45.0, "params_m": 12.0}]
    p = offline_propose(hist)
    assert p["name"] not in {"a", "b"}
    assert p["config"]["d_model"] % p["config"]["n_heads"] == 0
    assert "predicted" in p["rationale"]        # reasoned, not canned


def test_propose_respects_constraints_over_many_states():
    hist = []
    for i in range(15):
        p = offline_propose(hist)
        c = p["config"]
        assert c["d_model"] % c["n_heads"] == 0
        assert c["n_kv_heads"] in (1, 2, c["n_heads"])
        hist.append({"name": p["name"], "config": c,
                     "val_ppl": 20.0 + i, "params_m": 5.0})


# ── Evaluator ─────────────────────────────────────────────────────────────

def test_score_rewards_structure_penalizes_garbage():
    good = "ROMEO:\nBut soft what light through yonder window breaks\nJULIET:\nAy me"
    garbage = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12"
    qg = offline_score({"val_ppl": 20.0}, good)["sample_quality"]
    qb = offline_score({"val_ppl": 20.0}, garbage)["sample_quality"]
    assert qg > qb
    assert qg >= 7 and qb <= 4


def test_score_flags_divergence_in_notes():
    r = offline_score({"val_ppl": float("inf")}, "ROMEO:\nhello")
    assert not math.isfinite(r["val_ppl"])
    assert "diverg" in r["notes"].lower()


def test_score_empty_sample_is_worst():
    assert offline_score({"val_ppl": 20.0}, "")["sample_quality"] == 1


# ── Judge ──────────────────────────────────────────────────────────────────

def test_judge_accepts_clean_run():
    report = {"val_ppl": 22.0, "sample_quality": 8, "cloze_accuracy": 0.1,
              "notes": "ppl 22, cloze noted, legible"}
    v = offline_audit(_cfg(), report, "ROMEO:\ntext")
    assert v["accept"] and v["confidence"] > 0.5 and v["flagged"] == []


def test_judge_rejects_divergence():
    report = {"val_ppl": float("inf"), "sample_quality": 3,
              "cloze_accuracy": 0.0, "notes": "bad"}
    v = offline_audit(_cfg(), report, "junk")
    assert not v["accept"]
    assert any("diverg" in f for f in v["flagged"])


def test_judge_flags_quality_inflation():
    report = {"val_ppl": 120.0, "sample_quality": 9, "cloze_accuracy": 0.0,
              "notes": "looks good and coherent"}
    v = offline_audit(_cfg(), report, "ROMEO:\ntext")
    assert not v["accept"]
    assert any("inflated" in f for f in v["flagged"])


# ── MetaJudge ──────────────────────────────────────────────────────────────

def test_meta_detects_lenient_bias():
    hist = [{"verdict": {"accept": True}, "report": {"val_ppl": 110.0 + i}}
            for i in range(5)]
    m = offline_meta(hist, {"verdict": {"accept": True},
                            "report": {"val_ppl": 150.0}})
    assert m["bias_detected"] and "lenient" in m["bias_detected"]
    assert m["judge_was_correct"] is False       # accepted a poor run


def test_meta_flags_accepting_diverged_run():
    m = offline_meta([], {"verdict": {"accept": True},
                          "report": {"val_ppl": float("inf")}})
    assert m["judge_was_correct"] is False


def test_meta_passes_consistent_verdict():
    m = offline_meta([], {"verdict": {"accept": True},
                          "report": {"val_ppl": 21.0}})
    assert m["judge_was_correct"] is True
    assert m["bias_detected"] is None
