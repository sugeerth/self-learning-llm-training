"""Experiment sweep: train several variants, compare, pick a winner.

Each variant differs in ONE axis from the baseline so differences are
attributable. We run short (STEPS=400) but identical-length runs, use the
same seed, and measure: final val loss, perplexity, throughput, param count.

Winner = lowest val perplexity, with param count reported for fairness.
All results are written to `experiments.json` and surfaced on the dashboard.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict

import torch

from data import Loader, prepare
from model import LLM, ModelConfig
from train import build_optimizer, lr_at, pick_device

OUT = os.path.join(os.path.dirname(__file__), "experiments.json")
BEST_CKPT = os.path.join(os.path.dirname(__file__), "ckpt.pt")

# Short identical-length runs for fair comparison on a laptop.
STEPS = 200
WARMUP = 20
BLOCK = 128
MICRO_BATCH = 24
GRAD_ACCUM = 2
EVAL_ITERS = 20
SEED = 1337

VARIANTS = [
    ("baseline",
     ModelConfig(d_model=384, n_layers=6, n_heads=6, n_kv_heads=2, max_seq_len=BLOCK),
     {"lr": 3e-4}),
    ("no-GQA (full MHA)",
     ModelConfig(d_model=384, n_layers=6, n_heads=6, n_kv_heads=6, max_seq_len=BLOCK),
     {"lr": 3e-4}),
    ("shallow-wide (4L x 512d)",
     ModelConfig(d_model=512, n_layers=4, n_heads=8, n_kv_heads=2, max_seq_len=BLOCK),
     {"lr": 3e-4}),
    ("high-LR (6e-4)",
     ModelConfig(d_model=384, n_layers=6, n_heads=6, n_kv_heads=2, max_seq_len=BLOCK),
     {"lr": 6e-4}),
    ("untied embeddings",
     ModelConfig(d_model=384, n_layers=6, n_heads=6, n_kv_heads=2, max_seq_len=BLOCK, tie_embeddings=False),
     {"lr": 3e-4}),
    # --- new this run ---
    ("deep-narrow (10L x 320d)",
     ModelConfig(d_model=320, n_layers=10, n_heads=5, n_kv_heads=1, max_seq_len=BLOCK),
     {"lr": 3e-4}),
    ("combo: untied + high-LR",
     ModelConfig(d_model=384, n_layers=6, n_heads=6, n_kv_heads=2, max_seq_len=BLOCK, tie_embeddings=False),
     {"lr": 6e-4}),
    ("tiny (256d x 4L)",
     ModelConfig(d_model=256, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=BLOCK),
     {"lr": 6e-4}),
    ("long-context (seq=256)",
     ModelConfig(d_model=384, n_layers=6, n_heads=6, n_kv_heads=2, max_seq_len=256),
     {"lr": 3e-4}),
]


def lr_schedule(step: int, base_lr: float) -> float:
    return lr_local_schedule(step, base_lr, STEPS)


def lr_local_schedule(step: int, base_lr: float, total_steps: int) -> float:
    warmup = min(WARMUP, max(1, total_steps // 10))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total_steps:
        return base_lr * 0.1
    decay = (step - warmup) / max(1, total_steps - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * decay))
    return base_lr * 0.1 + coeff * (base_lr - base_lr * 0.1)


@torch.no_grad()
def eval_loss(model, loader, device, amp_dtype) -> float:
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = loader.batch()
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=device != "cpu"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def run_variant(name: str, cfg: ModelConfig, hp: dict, train_bin: str, val_bin: str,
                device: str, amp_dtype, seed: int = SEED, steps: int = STEPS,
                progress_cb=None) -> dict:
    torch.manual_seed(seed)
    seq = cfg.max_seq_len
    # keep tokens-per-step roughly constant when seq differs, for fair comparison
    bsz = max(1, MICRO_BATCH * BLOCK // seq)
    train_loader = Loader(train_bin, seq, bsz, device)
    val_loader = Loader(val_bin, seq, bsz, device)

    model = LLM(cfg).to(device)
    # hand-roll a light optimizer (same recipe as train.py)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
        lr=hp["lr"], betas=(0.9, 0.95), eps=1e-8)

    losses = []
    t0 = time.time()
    tokens = 0
    for step in range(steps):
        lr = lr_local_schedule(step, hp["lr"], steps)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = train_loader.batch()
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=device != "cpu"):
                _, loss = model(x, y)
            (loss / GRAD_ACCUM).backward()
            acc += loss.item() / GRAD_ACCUM
            tokens += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses.append(acc)
        if step % 50 == 0:
            print(f"  [{name}] step {step:3d}/{steps}  loss {acc:.4f}  lr {lr:.2e}")
        if progress_cb is not None and (step % 10 == 0 or step == steps - 1):
            progress_cb(step=step, total=steps, loss=acc, lr=lr)

    dt = time.time() - t0
    vloss = eval_loss(model, val_loader, device, amp_dtype)
    ppl = math.exp(vloss)
    return {
        "name": name,
        "config": asdict(cfg),
        "lr": hp["lr"],
        "params_m": round(model.num_params() / 1e6, 2),
        "train_loss_final": losses[-1],
        "val_loss": vloss,
        "val_ppl": ppl,
        "elapsed_s": round(dt, 1),
        "tokens_seen": tokens,
        "tok_per_sec": round(tokens / dt, 1),
        "loss_curve": losses[::5],  # decimate
        "model": model,  # kept so we can save the winner
    }


def _stats(xs: list[float]) -> dict:
    n = len(xs)
    mean = sum(xs) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in xs) / n if n else 0.0
    return {"mean": mean, "std": var ** 0.5, "n": n}


def run_custom(name: str, cfg: ModelConfig, lr: float, seeds: list[int],
               steps: int = STEPS, progress_cb=None) -> dict:
    """Run one model config across multiple seeds. Returns aggregate mean±std.

    progress_cb(stage, **kwargs) — stage ∈ {"seed_start","step","seed_done","all_done"}.
    """
    device, amp_dtype = pick_device()
    train_bin, val_bin = prepare()
    per_seed = []
    for i, seed in enumerate(seeds):
        if progress_cb:
            progress_cb(stage="seed_start", seed=seed, seed_idx=i, n_seeds=len(seeds))

        def _step_cb(step, total, loss, lr):
            if progress_cb:
                progress_cb(stage="step", seed=seed, seed_idx=i, n_seeds=len(seeds),
                            step=step, total=total, loss=loss, lr=lr)

        r = run_variant(
            f"{name} (seed={seed})", cfg, {"lr": lr}, train_bin, val_bin,
            device, amp_dtype, seed=seed, steps=steps, progress_cb=_step_cb,
        )
        r.pop("model", None)
        per_seed.append(r)
        if progress_cb:
            progress_cb(stage="seed_done", seed=seed, seed_idx=i, n_seeds=len(seeds),
                        val_loss=r["val_loss"], val_ppl=r["val_ppl"])

    # aggregate
    agg = {
        "name": name,
        "config": asdict(cfg),
        "lr": lr,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "params_m": per_seed[0]["params_m"],
        "steps": steps,
        "val_loss": _stats([r["val_loss"] for r in per_seed]),
        "val_ppl": _stats([r["val_ppl"] for r in per_seed]),
        "train_loss_final": _stats([r["train_loss_final"] for r in per_seed]),
        "tok_per_sec": _stats([r["tok_per_sec"] for r in per_seed]),
        "elapsed_s": round(sum(r["elapsed_s"] for r in per_seed), 1),
        # use the mean curve (per-step averaged across seeds) for the chart
        "loss_curve": [
            sum(r["loss_curve"][k] for r in per_seed) / len(per_seed)
            for k in range(min(len(r["loss_curve"]) for r in per_seed))
        ],
        "per_seed": [
            {k: v for k, v in r.items() if k in ("val_loss", "val_ppl", "train_loss_final", "loss_curve")}
            for r in per_seed
        ],
    }
    if progress_cb:
        progress_cb(stage="all_done", agg=agg)
    return agg


def append_result(agg: dict, path: str = OUT) -> None:
    """Append a multi-seed result to experiments.json (preserving prior variants)."""
    data = {"variants": []}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            pass
    # flatten multi-seed agg into a variant entry the dashboard can already render
    flat = {
        "name": agg["name"],
        "config": agg["config"],
        "lr": agg["lr"],
        "params_m": agg["params_m"],
        "train_loss_final": agg["train_loss_final"]["mean"],
        "val_loss": agg["val_loss"]["mean"],
        "val_loss_std": agg["val_loss"]["std"],
        "val_ppl": agg["val_ppl"]["mean"],
        "val_ppl_std": agg["val_ppl"]["std"],
        "tok_per_sec": agg["tok_per_sec"]["mean"],
        "elapsed_s": agg["elapsed_s"],
        "loss_curve": agg["loss_curve"],
        "n_seeds": agg["n_seeds"],
        "seeds": agg["seeds"],
        "steps": agg["steps"],
        "custom": True,
    }
    # replace if a custom run with the same name already exists
    variants = [v for v in data.get("variants", []) if v.get("name") != flat["name"]]
    variants.append(flat)
    variants.sort(key=lambda v: v.get("val_loss", float("inf")))
    data["variants"] = variants
    best = variants[0]
    data["winner"] = best["name"]
    data["why"] = (
        f"{best['name']}: val_loss={best['val_loss']:.4f} "
        f"(ppl {best['val_ppl']:.2f}), {best.get('n_seeds',1)} seed(s), "
        f"{best['params_m']}M params."
    )
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    device, amp_dtype = pick_device()
    print(f"Device: {device}  bf16: {amp_dtype == torch.bfloat16}")
    train_bin, val_bin = prepare()

    results = []
    for name, cfg, hp in VARIANTS:
        print(f"\n=== {name} ===")
        r = run_variant(name, cfg, hp, train_bin, val_bin, device, amp_dtype)
        print(f"  → val_loss={r['val_loss']:.4f}  ppl={r['val_ppl']:.2f}  "
              f"params={r['params_m']}M  {r['tok_per_sec']:.0f} tok/s")
        results.append(r)

    # rank by val loss
    results.sort(key=lambda r: r["val_loss"])
    best = results[0]
    print(f"\n🏆 Winner: {best['name']}  (val_loss={best['val_loss']:.4f}  ppl={best['val_ppl']:.2f})")

    # save best ckpt so benchmark.py / server.py can load it
    torch.save({"model": best["model"].state_dict(),
                "cfg": best["config"]}, BEST_CKPT)

    # write results (strip model objects for JSON)
    serializable = [{k: v for k, v in r.items() if k != "model"} for r in results]
    with open(OUT, "w") as f:
        json.dump({
            "variants": serializable,
            "winner": best["name"],
            "why": (
                f"{best['name']} achieved the lowest validation loss "
                f"({best['val_loss']:.4f}, ppl {best['val_ppl']:.2f}) at "
                f"{best['params_m']}M params and {best['tok_per_sec']:.0f} tok/s."
            ),
            "steps_per_variant": STEPS,
            "block_size": BLOCK,
        }, f, indent=2)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
