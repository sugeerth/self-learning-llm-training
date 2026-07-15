"""Offline agent brains — the multi-agent loop with no API key.

Each function is a real, deterministic decision-maker that mirrors what the
corresponding LLM agent does, using the same inputs and returning the same
structured shape. No network, no ANTHROPIC_API_KEY, no torch. The loop stays
genuinely *agentic* — the Trainer explores/exploits the search space, the
Evaluator scores samples from real signals, the Judge audits the exact
failure modes it's told to look for, and the MetaJudge detects bias
statistically over the verdict history.

agents.py routes to these whenever the key is absent (or AGENTS_OFFLINE=1).
"""

from __future__ import annotations

import hashlib
import math
import random
import re

from hyperband import CheapPrior, random_config

# search space (mirrors hyperband.random_config)
_D_MODELS = [128, 192, 256, 320, 384, 448, 512, 640]
_N_LAYERS = [2, 4, 6, 8, 10]
_FF_MULTS = [2.0, 8 / 3, 3.0, 4.0]


def _seeded_rng(*parts) -> random.Random:
    """Deterministic RNG from the inputs — same state proposes the same move,
    which keeps the loop reproducible without a fixed global seed."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# ────────────────────────────── Trainer ──────────────────────────────

def _mutate(cfg: dict, rng: random.Random) -> dict:
    """One coordinated step in the search space, respecting the head/kv
    divisibility constraints so the config is always trainable."""
    c = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    dim = rng.choice(["d_model", "n_layers", "d_ff_mult", "n_kv_heads"])
    if dim == "d_model":
        c["d_model"] = rng.choice(_D_MODELS)
        heads = [h for h in (2, 4, 6, 8) if c["d_model"] % h == 0]
        c["n_heads"] = rng.choice(heads)
        c["n_kv_heads"] = rng.choice([1, 2, c["n_heads"]])
    elif dim == "n_layers":
        c["n_layers"] = rng.choice(_N_LAYERS)
    elif dim == "d_ff_mult":
        c["d_ff_mult"] = rng.choice(_FF_MULTS)
    else:
        heads = c.get("n_heads", 4)
        c["n_kv_heads"] = rng.choice([1, 2, heads])
    return c


def offline_propose(history: list[dict], prior_path: str = "prior_store.json") -> dict:
    """Explore-exploit proposal.

    Cold start -> a random arch. Otherwise: build a candidate pool from
    mutations of the current best plus fresh random draws, rank them by the
    CheapPrior's acquisition (lower predicted log-ppl is better), and pick the
    most attractive arch not already tried. The prior is loaded from disk, so
    this compounds across runs exactly like the online path.
    """
    names = {h.get("name") for h in history}
    rng = _seeded_rng("trainer", len(history),
                      history[0].get("val_ppl") if history else "cold")

    if not history:
        cfg = random_config(rng)
        return {"name": f"cold-{rng.randint(1000, 9999)}", "config": cfg,
                "lr": 3e-4,
                "rationale": "cold start: no priors yet, sampling the space."}

    prior = CheapPrior.load(prior_path)
    for h in history:                       # fold visible history into the prior
        if h.get("config") and h.get("val_ppl"):
            prior.add(h["config"], float(h["val_ppl"]))

    best = min(history, key=lambda h: h.get("val_ppl", float("inf")))
    best_cfg = best["config"]

    # candidate pool: exploit (mutate best) + explore (random draws)
    pool = [_mutate(best_cfg, rng) for _ in range(12)]
    pool += [random_config(rng) for _ in range(12)]
    # rank by acquisition (lower = more attractive); prior always answers
    pool.sort(key=lambda c: prior.acquisition(c))

    chosen = pool[0]
    mean, unc = prior.predict(chosen)
    is_mutation = any(chosen.get(k) == best_cfg.get(k)
                      for k in ("d_model", "n_layers"))
    kind = "exploit near best" if is_mutation else "explore new region"
    n = 0
    name = f"off-{len(history)}-{rng.randint(100, 999)}"
    while name in names and n < 10:         # avoid duplicate names
        name = f"off-{len(history)}-{rng.randint(100, 999)}"
        n += 1
    return {
        "name": name, "config": chosen, "lr": 3e-4,
        "rationale": (f"{kind}: d_model={chosen['d_model']} "
                      f"n_layers={chosen['n_layers']}; predicted "
                      f"ppl~{math.e ** mean:.0f} (unc {unc:.2f}) vs best "
                      f"{best.get('val_ppl', 0):.0f}."),
    }


# ────────────────────────────── Evaluator ──────────────────────────────

def _sample_quality(sample: str) -> tuple[int, list[str]]:
    """1-10 from real signals: printable ratio, repetition, and whether the
    text carries Shakespeare-play structure (speaker lines, colons). NOT
    semantics — nonsense is expected at low step counts."""
    obs: list[str] = []
    s = sample or ""
    if len(s.strip()) < 5:
        return 1, ["empty/near-empty sample"]

    printable = sum(c.isalpha() or c.isspace() or c in ".,:;!?'-" for c in s)
    r_print = printable / len(s)

    bigrams = [s[i:i + 2] for i in range(len(s) - 1)]
    r_unique = len(set(bigrams)) / max(len(bigrams), 1)      # low => repetitive

    has_speaker = bool(re.search(r"[A-Z]{2,}:", s))          # "ROMEO:"
    has_lines = "\n" in s.strip()

    score = 1.0
    score += 4.0 * r_print                    # legible characters
    score += 3.0 * min(r_unique * 1.5, 1.0)   # non-repetitive
    score += 1.0 if has_speaker else 0.0      # play structure
    score += 1.0 if has_lines else 0.0
    q = max(1, min(10, round(score)))

    if r_print < 0.6:
        obs.append("many non-text bytes")
    if r_unique < 0.35:
        obs.append("highly repetitive")
    if has_speaker:
        obs.append("has speaker-line structure")
    if not obs:
        obs.append("legible, varied")
    return q, obs


def offline_score(raw_metrics: dict, sample: str) -> dict:
    q, obs = _sample_quality(sample)
    val_ppl = float(raw_metrics.get("val_ppl", float("inf")))
    note = f"ppl={val_ppl:.1f}; sample: {', '.join(obs)}."
    if not math.isfinite(val_ppl):
        note = "diverged (non-finite val_ppl); " + note
    return {
        "val_loss": float(raw_metrics.get("val_loss", 0.0)),
        "val_ppl": val_ppl,
        "cloze_accuracy": float(raw_metrics.get("cloze_accuracy", 0.0)),
        "sample_quality": q,
        "notes": note[:200],
    }


# ────────────────────────────── Judge ──────────────────────────────

def offline_audit(proposal: dict, report: dict, sample: str) -> dict:
    """Skeptical rule-based audit of the evaluator's report — the exact
    failure modes the JudgeAgent prompt lists."""
    flagged: list[str] = []
    ppl = float(report.get("val_ppl", float("inf")))
    q = int(report.get("sample_quality", 0))
    notes = (report.get("notes") or "").lower()

    if not math.isfinite(ppl):
        if "diverg" not in notes:
            flagged.append("divergence not noted")
    else:
        # quality inflated relative to a poor ppl
        if q >= 7 and ppl > 60:
            flagged.append("sample_quality inflated relative to val_ppl")
        # optimistic notes contradicting bad numbers
        if ppl > 80 and any(w in notes for w in ("good", "coherent", "strong")):
            flagged.append("notes contradict the numbers")
    if report.get("cloze_accuracy", 0) == 0 and "cloze" not in notes:
        flagged.append("cloze_accuracy ignored")

    serious = [f for f in flagged if "ignored" not in f]
    accept = math.isfinite(ppl) and not serious
    # confidence: high when the picture is clear-cut, lower near the margin
    if not math.isfinite(ppl):
        conf = 0.95
    elif not flagged:
        conf = 0.7 + min(0.25, 20.0 / max(ppl, 1))
    else:
        conf = 0.5 - 0.1 * len(serious)
    conf = round(max(0.1, min(0.99, conf)), 2)

    if accept and not flagged:
        reason = f"metrics consistent; ppl {ppl:.0f}, quality {q}/10. Accept."
    elif accept:
        reason = f"minor flags ({len(flagged)}) but no disqualifier; accept with reservations."
    else:
        reason = f"reject: {'; '.join(flagged) or 'non-finite ppl'}."
    return {"accept": bool(accept), "confidence": conf,
            "flagged": flagged, "reasoning": reason[:300]}


# ────────────────────────────── MetaJudge ──────────────────────────────

def offline_meta(judge_history: list[dict], current: dict) -> dict:
    """Detect systematic judge bias over the verdict history, and sanity-check
    the current verdict against the run's own numbers."""
    verdicts = [h.get("verdict", {}) for h in judge_history if h.get("verdict")]
    reports = [h.get("report", {}) for h in judge_history if h.get("report")]

    bias = None
    if len(verdicts) >= 4:
        accepts = [v.get("accept") for v in verdicts]
        rate = sum(bool(a) for a in accepts) / len(accepts)
        # lenient: accepts runs with poor ppl
        accepted_bad = [r for v, r in zip(verdicts, reports)
                        if v.get("accept") and r.get("val_ppl", 0) > 80]
        rejected_ok = [r for v, r in zip(verdicts, reports)
                       if not v.get("accept") and 0 < r.get("val_ppl", 1e9) < 40]
        if rate > 0.8 and len(accepted_bad) >= 2:
            bias = "lenient: accepts high-ppl runs that should be rejected"
        elif rate < 0.2 and len(rejected_ok) >= 2:
            bias = "strict: rejects low-ppl runs matching accept patterns"

    # was THIS verdict correct? check it against the run's own report
    cur_v = current.get("verdict", {})
    cur_r = current.get("report", {})
    ppl = float(cur_r.get("val_ppl", float("inf")))
    accept = bool(cur_v.get("accept"))
    correct = True
    why = "verdict consistent with the run's metrics."
    if accept and not math.isfinite(ppl):
        correct, why = False, "accepted a diverged run (non-finite ppl)."
    elif accept and ppl > 100:
        correct, why = False, f"accepted a very poor run (ppl {ppl:.0f})."
    elif not accept and 0 < ppl < 30:
        correct, why = False, f"rejected a strong run (ppl {ppl:.0f})."
    if bias:
        why = f"{why} History suggests {bias.split(':')[0]} bias."
    return {"judge_was_correct": correct, "bias_detected": bias,
            "reasoning": why[:300]}
