"""Flywheel experiment — measure whether self-generated data helps or hurts.

The self-learning claim (10X_PLAN item 3): winner model generates data ->
filter -> mix into the next round's training set -> better model. This driver
turns that claim into a controlled, paired A/B:

  1. Train a GENERATOR through the harness.
  2. Generate synthetic continuations from real-prefix seeds (token space).
  3. FILTER: offline heuristic gates (degenerate-repetition, low-diversity,
     near-duplicate) — works air-gapped; with ANTHROPIC_API_KEY set, the
     3-judge Claude ensemble from synthetic_flywheel.py takes over.
  4. Mix survivors into a real-data subset at each ratio r.
  5. Train IDENTICALLY-SEEDED models on real-only vs mixed corpora
     (same config, same steps, same init) and compare deterministic val
     loss on the REAL validation set.

Verdict per ratio: gain / neutral / collapse, mean over seeds.

CLI:
    python3 flywheel.py run --quick    # ~10 min on 4 CPU cores
    python3 flywheel.py run            # more seeds, samples, ratios

Outputs: flywheel_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np

REPORT = os.path.join(os.path.dirname(__file__), "flywheel_report.json")

# generator + A/B trainee share this config (small = fast on CPU)
GEN_CFG = {"vocab_size": 50304, "d_model": 256, "n_layers": 4, "n_heads": 4,
           "n_kv_heads": 2, "d_ff_mult": 8 / 3, "max_seq_len": 128,
           "rope_theta": 10000.0, "dropout": 0.0, "tie_embeddings": True}


# ────────────────────── offline filters (pure python — unit-tested) ──────────────────────

def repetition_fraction(tokens: list[int], n: int = 4) -> float:
    """Fraction of n-grams that are repeats. Degenerate loops score near 1."""
    if len(tokens) < n + 1:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def distinct_ratio(tokens: list[int], n: int = 2) -> float:
    """Distinct n-grams / total n-grams. Low = the sample keeps saying the
    same thing."""
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def dedup(samples: list[list[int]], n: int = 8) -> list[list[int]]:
    """Drop samples sharing any n-gram with an earlier-kept sample."""
    seen: set = set()
    kept = []
    for s in samples:
        grams = {tuple(s[i:i + n]) for i in range(max(len(s) - n + 1, 0))}
        if grams & seen:
            continue
        seen |= grams
        kept.append(s)
    return kept


def heuristic_filter(samples: list[list[int]],
                     max_repetition: float = 0.5,
                     min_distinct: float = 0.35) -> tuple[list[list[int]], dict]:
    """Offline gate chain. Returns (kept, stats)."""
    n0 = len(samples)
    s1 = [s for s in samples if repetition_fraction(s) <= max_repetition]
    s2 = [s for s in s1 if distinct_ratio(s) >= min_distinct]
    s3 = dedup(s2)
    stats = {"generated": n0, "after_repetition_gate": len(s1),
             "after_diversity_gate": len(s2), "after_dedup": len(s3),
             "kept_rate": round(len(s3) / max(n0, 1), 3)}
    return s3, stats


def judge_filter(samples: list[list[int]]) -> tuple[list[list[int]], dict]:
    """Claude 3-judge ensemble (synthetic_flywheel.py) — only with a key."""
    from data import tokenizer
    from synthetic_flywheel import CoarseJudge, FineJudge, JudgeOfJudge
    enc = tokenizer()
    coarse, fine, jojo = CoarseJudge(), FineJudge(), JudgeOfJudge()
    kept = []
    for s in samples:
        text = enc.decode(s)
        c = coarse.classify(text)
        if not c.get("is_shakespearean"):
            continue
        f = fine.score(text)
        if f.get("score", 0) < 7:
            continue
        jj = jojo.audit(text, f)
        if jj.get("score_calibrated"):
            kept.append(s)
    return kept, {"generated": len(samples), "kept": len(kept),
                  "kept_rate": round(len(kept) / max(len(samples), 1), 3),
                  "filter": "claude-3-judge"}


def mix_corpus(real: np.ndarray, samples: list[list[int]], ratio: float,
               out_path: str, sep_token: int = 10) -> dict:
    """real subset + synthetic tokens capped at ratio*len(real) -> bin file."""
    synth: list[int] = []
    cap = int(len(real) * ratio)
    for s in samples:
        if len(synth) >= cap:
            break
        synth.extend(s[: cap - len(synth)])
        synth.append(sep_token)
    mixed = np.concatenate([real, np.asarray(synth, dtype=np.uint16)])
    mixed.tofile(out_path)
    return {"real_tokens": int(len(real)), "synth_tokens": len(synth),
            "ratio_actual": round(len(synth) / len(real), 4), "path": out_path}


# ────────────────────── generation (token space, batched) ──────────────────────

def generate_samples(ckpt_path: str, n_samples: int, sample_len: int,
                     prefix_len: int, seed: int, batch: int = 32) -> list[list[int]]:
    """Batched continuations from random REAL train prefixes; returns only the
    generated part (never re-adds real tokens to the mix)."""
    import torch
    from data import prepare
    from harness import _clean, _device
    from model import LLM, ModelConfig

    device = _device()
    model = LLM(ModelConfig(**_clean(GEN_CFG))).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    train_bin, _ = prepare()
    data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    rng = random.Random(seed)
    torch.manual_seed(seed)

    out: list[list[int]] = []
    while len(out) < n_samples:
        b = min(batch, n_samples - len(out))
        starts = [rng.randrange(0, len(data) - prefix_len - 1) for _ in range(b)]
        prefixes = torch.stack([
            torch.from_numpy(data[i:i + prefix_len].astype("int64")) for i in starts
        ]).to(device)
        gen = model.generate(prefixes, max_new_tokens=sample_len, temperature=0.9, top_k=50)
        out.extend(row[prefix_len:].tolist() for row in gen.cpu())
    return out


# ────────────────────── the A/B experiment ──────────────────────

def run_experiment(gen_steps: int, n_samples: int, sample_len: int,
                   real_tokens: int, ratios: list[float], seeds: int,
                   ab_steps: int) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from data import prepare
    from harness import RUNS_DIR, _train_task, _worker_init, load_or_tune

    profile = load_or_tune(quick=True)
    run_dir = os.path.join(RUNS_DIR, f"flywheel-{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    train_bin, _ = prepare()

    base_task = {"lr": 3e-4, "batch_size": profile.batch_size, "val_batches": 8,
                 "ckpt_in": None, "ckpt_out": None, "want_sample": False,
                 "deterministic_val": True}

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=profile.workers, mp_context=ctx,
                             initializer=_worker_init,
                             initargs=(profile.threads_per_worker,)) as pool:
        # 1. generator
        gen_ck = os.path.join(run_dir, "generator.pt")
        t0 = time.time()
        gen_ev = pool.submit(_train_task, dict(
            base_task, cfg=GEN_CFG, steps=gen_steps, ckpt_out=gen_ck,
            torch_seed=1234, threads=os.cpu_count())).result()
        print(f"flywheel: generator trained {gen_steps} steps, "
              f"val_ppl={gen_ev['val_ppl']:.1f} ({time.time() - t0:.0f}s)")

        # 2. generate + 3. filter
        t0 = time.time()
        raw = generate_samples(gen_ck, n_samples, sample_len, prefix_len=16, seed=7)
        if os.environ.get("ANTHROPIC_API_KEY"):
            kept, fstats = judge_filter(raw)
        else:
            kept, fstats = heuristic_filter(raw)
        print(f"flywheel: {fstats} ({time.time() - t0:.0f}s)")

        # 4. corpora — one shared real subset, one mixed bin per ratio
        real = np.array(np.memmap(train_bin, dtype=np.uint16, mode="r")[:real_tokens])
        real_path = os.path.join(run_dir, "real.bin")
        real.tofile(real_path)
        mixes = {}
        for r in ratios:
            path = os.path.join(run_dir, f"mixed_{r}.bin")
            mixes[r] = mix_corpus(real, kept, r, path)
            print(f"flywheel: ratio {r}: {mixes[r]}")

        # 5. paired A/B — identical torch_seed per (seed, variant) pair
        variants = [("real-only", real_path)] + [(f"mix@{r}", mixes[r]["path"]) for r in ratios]
        tasks, keys = [], []
        for seed in range(seeds):
            for name, path in variants:
                keys.append((seed, name))
                tasks.append(dict(base_task, cfg=GEN_CFG, steps=ab_steps,
                                  train_bin=path, torch_seed=1000 + seed))
        t0 = time.time()
        evals = list(pool.map(_train_task, tasks))
        print(f"flywheel: {len(tasks)} A/B trainings done ({time.time() - t0:.0f}s)")

    # 6. verdicts
    by_variant: dict[str, list[float]] = {}
    for (seed, name), ev in zip(keys, evals):
        by_variant.setdefault(name, []).append(ev["val_loss"])
    control = by_variant["real-only"]
    results = {}
    for name, losses in by_variant.items():
        deltas = [l - c for l, c in zip(losses, control)]
        mean_d = sum(deltas) / len(deltas)
        rel = mean_d / (sum(control) / len(control))
        verdict = ("control" if name == "real-only" else
                   "gain" if rel < -0.02 else "collapse" if rel > 0.02 else "neutral")
        results[name] = {"val_loss_per_seed": [round(l, 4) for l in losses],
                         "delta_vs_control": round(mean_d, 4),
                         "relative": f"{rel * +100:+.2f}%", "verdict": verdict}

    report = {"generator": {"steps": gen_steps, "val_ppl": round(gen_ev["val_ppl"], 2)},
              "filter": fstats, "mixes": {str(k): v for k, v in mixes.items()},
              "ab_steps": ab_steps, "seeds": seeds, "results": results}
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nflywheel: A/B at {ab_steps} steps x {seeds} seeds (val loss on REAL val, lower=better)")
    for name, r in results.items():
        print(f"  {name:<12} loss={[f'{l:.3f}' for l in r['val_loss_per_seed']]}"
              f"  vs control {r['relative']:>7}  -> {r['verdict']}")
    print(f"flywheel: wrote {REPORT}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the flywheel A/B experiment")
    r.add_argument("--quick", action="store_true", help="small everything (~10 min CPU)")
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--gen-steps", type=int, default=200)
    r.add_argument("--ab-steps", type=int, default=96)
    r.add_argument("--samples", type=int, default=600)
    r.add_argument("--ratios", type=float, nargs="+", default=[0.05, 0.2])
    args = ap.parse_args()

    if args.cmd == "run":
        kw = dict(gen_steps=args.gen_steps, n_samples=args.samples, sample_len=64,
                  real_tokens=120_000, ratios=args.ratios, seeds=args.seeds,
                  ab_steps=args.ab_steps)
        if args.quick:
            kw.update(gen_steps=96, n_samples=384, seeds=2, ab_steps=64)
        run_experiment(**kw)


if __name__ == "__main__":
    main()
