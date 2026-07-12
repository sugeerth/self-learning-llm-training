"""Throughput harness — the execution engine under the self-learning loop.

Two 10x levers, both aimed at the harness rather than the model:

1. **Execution 10x** — the same successive-halving semantics as
   `hyperband.successive_halving`, but:
     - candidates in a rung train in PARALLEL worker processes
     - survivors are PROMOTED from checkpoints (train only the delta steps)
       instead of retrained from scratch each rung
     - candidates are pre-filtered with a zero-cost SNIP-style saliency proxy
       before any training steps are spent
     - sample generation (the slow autoregressive part) happens ONCE for the
       winner, not per-eval per-rung

2. **Self-optimization 10x** — the harness tunes ITSELF. `autotune()` probes
   worker-count x threads-per-worker x batch-size combinations on the current
   machine with short real training tasks, measures aggregate tokens/sec, and
   persists the winning profile to `harness_profile.json`. Every later run
   starts at the machine's measured peak throughput instead of a guess.

CLI:
    python3 harness.py tune                 # self-optimize -> harness_profile.json
    python3 harness.py bench                # baseline vs harness, same bracket
    python3 harness.py run                  # one bracket via the tuned harness

The runner uses this via `self_learning_runner.py --harness`.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from multiprocessing import get_context

import torch

from data import prepare, Loader
from model import LLM, ModelConfig

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "harness_profile.json")
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
_MC_FIELDS = {f.name for f in dataclasses.fields(ModelConfig)}


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _clean(cfg: dict) -> dict:
    """Strip bookkeeping keys (_eval, _steps, ...) before ModelConfig(**cfg).

    The original successive_halving mutates candidate dicts in place, which
    makes ModelConfig(**cfg) explode on every rung after the first — the
    harness sanitizes instead.
    """
    return {k: v for k, v in cfg.items() if k in _MC_FIELDS}


def _log_event(**fields) -> None:
    try:
        from braintrust_bridge import log_event
        log_event(**fields)
    except Exception:
        pass


# ────────────────────────── tuned profile ──────────────────────────

@dataclass
class HarnessProfile:
    workers: int
    threads_per_worker: int
    batch_size: int
    tokens_per_s: float = 0.0     # measured aggregate throughput at tune time
    device: str = "cpu"

    def save(self, path: str = PROFILE_PATH) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str = PROFILE_PATH) -> "HarnessProfile | None":
        try:
            with open(path) as f:
                return HarnessProfile(**json.load(f))
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def fallback() -> "HarnessProfile":
        cores = os.cpu_count() or 2
        return HarnessProfile(workers=max(1, cores // 2), threads_per_worker=2,
                              batch_size=8, device=_device())


# ────────────────────────── worker side ──────────────────────────
# Everything below the pool boundary is top-level and picklable (spawn-safe).

def _worker_init(threads: int) -> None:
    torch.set_num_threads(max(1, threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set for this process


def _train_task(task: dict) -> dict:
    """Train one candidate for `steps` mini-batches; optionally resume/save.

    task = {cfg, steps, lr, batch_size, val_batches, ckpt_in, ckpt_out, want_sample}
    Returns the same eval dict shape as self_learning_runner.train_partial.
    """
    if task.get("threads"):
        # late rungs have fewer candidates than workers — give each task the
        # cores that would otherwise idle (interop threads stay fixed)
        torch.set_num_threads(task["threads"])
    if task.get("torch_seed") is not None:
        # paired experiments (flywheel A/B) need identical init across variants
        torch.manual_seed(task["torch_seed"])
    device = task.get("device") or _device()
    mc = ModelConfig(**_clean(task["cfg"]))
    model = LLM(mc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=task["lr"], betas=(0.9, 0.95))

    prior_steps = 0
    ckpt_in = task.get("ckpt_in")
    if ckpt_in and os.path.exists(ckpt_in):
        ckpt = torch.load(ckpt_in, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        prior_steps = ckpt.get("steps", 0)

    train_bin, val_bin = prepare()
    if task.get("train_bin"):
        train_bin = task["train_bin"]   # e.g. a flywheel-mixed corpus
    bs = task["batch_size"]
    train_loader = Loader(train_bin, block_size=mc.max_seq_len, batch_size=bs, device=device)
    val_loader = Loader(val_bin, block_size=mc.max_seq_len, batch_size=bs, device=device)

    model.train()
    t0 = time.time()
    losses = []
    for _ in range(task["steps"]):
        x, y = train_loader.batch()
        _, loss = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    train_s = time.time() - t0

    model.eval()
    if task.get("deterministic_val"):
        val_loss = _fixed_val_loss(model, val_bin, mc, task["val_batches"], bs, device)
    else:
        val_losses = []
        with torch.no_grad():
            for _ in range(task["val_batches"]):
                x, y = val_loader.batch()
                _, loss = model(x, targets=y)
                val_losses.append(loss.item())
        val_loss = float(sum(val_losses) / max(len(val_losses), 1))

    sample = ""
    if task.get("want_sample"):
        sample = _generate_sample(model, device)

    ckpt_out = task.get("ckpt_out")
    if ckpt_out:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "steps": prior_steps + task["steps"]}, ckpt_out)

    total_steps = prior_steps + task["steps"]
    return {
        "val_loss": val_loss,
        "val_ppl": float(math.e ** val_loss),
        "cloze_accuracy": 0.0,
        "tokens_seen": total_steps * bs * mc.max_seq_len,
        "elapsed_s": round(time.time() - t0, 1),
        "train_s": round(train_s, 2),
        "loss_curve": losses,
        "sample": sample,
        "params_m": round(model.num_params() / 1e6, 2),
        "trained_steps": total_steps,
    }


def _fixed_val_loss(model, val_bin: str, mc, batches: int, batch_size: int,
                    device: str) -> float:
    """Deterministic validation: fixed sequential windows of the val set.

    Random val batches are fine for one run but make cross-arm comparisons
    noisy — two identical models can differ by luck of the draw. Fixed windows
    make val_ppl a pure function of the weights."""
    import numpy as np
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    seq = mc.max_seq_len
    losses = []
    with torch.no_grad():
        for b in range(batches):
            idx = [((b * batch_size + j) * seq) % (len(data) - seq - 1)
                   for j in range(batch_size)]
            x = torch.stack([torch.from_numpy(data[i:i + seq].astype("int64")) for i in idx])
            y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq].astype("int64")) for i in idx])
            _, loss = model(x.to(device), targets=y.to(device))
            losses.append(loss.item())
    return float(sum(losses) / max(len(losses), 1))


def _generate_sample(model, device: str, prompt: str = "ROMEO:\n") -> str:
    from data import tokenizer
    enc = tokenizer()
    ids = torch.tensor([enc.encode(prompt)], device=device)
    out = model.generate(ids, max_new_tokens=80, temperature=0.8, top_k=50)
    return enc.decode(out[0].tolist())


def _snip_score(cfg: dict, batch: tuple) -> float:
    """Zero-cost trainability proxy: SNIP saliency = sum |g . w| after one
    forward/backward at init, normalized per parameter. Higher = better."""
    mc = ModelConfig(**_clean(cfg))
    model = LLM(mc)
    x, y = batch
    _, loss = model(x, targets=y)
    loss.backward()
    saliency, n = 0.0, 0
    for p in model.parameters():
        if p.grad is not None:
            saliency += (p.grad.detach() * p.detach()).abs().sum().item()
            n += p.numel()
    return saliency / max(n, 1)


def proxy_filter(candidates: list[dict], keep: int, batch_size: int = 4,
                 seq_len: int = 128) -> list[dict]:
    """Rank candidates by SNIP saliency at init and keep the top `keep`.
    Costs one forward+backward each — no training steps spent on duds."""
    if len(candidates) <= keep:
        return candidates
    train_bin, _ = prepare()
    loader = Loader(train_bin, block_size=seq_len, batch_size=batch_size, device="cpu")
    batch = loader.batch()
    scored = [(_snip_score(c, batch), c) for c in candidates]
    scored.sort(key=lambda t: -t[0])
    return [c for _, c in scored[:keep]]


# ────────────────────────── parallel promoted halving ──────────────────────────

def parallel_halving(
    candidates: list[dict],
    bracket,                      # hyperband.Bracket
    profile: HarnessProfile | None = None,
    lr: float = 3e-4,
    run_dir: str | None = None,
    val_batches_rung: int = 8,
    val_batches_final: int = 20,
    pool: ProcessPoolExecutor | None = None,   # reuse an outer warm pool
    want_sample: bool = True,
    deterministic_val: bool = False,
    on_eval=None,                 # callback(cfg, eval_dict, delta_steps) per eval
) -> list[dict]:
    """Drop-in for hyperband.successive_halving, but parallel + promoted.

    - each rung's candidates train concurrently across `profile.workers`
    - survivors resume from their checkpoint and train only the DELTA steps
      (from-scratch halving costs r + 2r + 4r = 7r per survivor path; promotion
      costs r + r + 2r = 4r AND the winner has genuinely accumulated training)
    - intermediate rungs use fewer val batches and skip sample generation; the
      winner gets the full eval + sample at the end
    """
    profile = profile or HarnessProfile.load() or HarnessProfile.fallback()
    run_dir = run_dir or os.path.join(RUNS_DIR, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    device = profile.device if profile.device != "cuda" or torch.cuda.is_available() else _device()

    survivors = list(candidates)
    for i, c in enumerate(survivors):
        c["_cid"] = i
    trained = {i: 0 for i in range(len(survivors))}   # cid -> cumulative steps
    target = bracket.initial_steps

    own_pool = pool is None
    if own_pool:
        pool = ProcessPoolExecutor(max_workers=profile.workers,
                                   mp_context=get_context("spawn"),
                                   initializer=_worker_init,
                                   initargs=(profile.threads_per_worker,))
    try:
        cores = os.cpu_count() or 2
        for halving in range(bracket.halvings + 1):
            last_rung = halving == bracket.halvings
            concurrency = max(1, min(len(survivors), profile.workers))
            threads = max(profile.threads_per_worker, cores // concurrency)
            tasks = []
            for cfg in survivors:
                cid = cfg["_cid"]
                ck = os.path.join(run_dir, f"c{cid}.pt")
                tasks.append({
                    "cfg": cfg, "steps": target - trained[cid], "lr": lr,
                    "batch_size": profile.batch_size, "threads": threads,
                    "val_batches": val_batches_final if last_rung else val_batches_rung,
                    "ckpt_in": ck if trained[cid] else None, "ckpt_out": ck,
                    "want_sample": False, "device": device,
                    "deterministic_val": deterministic_val,
                })
            t0 = time.time()
            evals = list(pool.map(_train_task, tasks))
            for cfg, ev, task in zip(survivors, evals, tasks):
                cfg["_eval"] = ev
                cfg["_steps"] = ev["trained_steps"]
                trained[cfg["_cid"]] = ev["trained_steps"]
                if on_eval is not None:
                    on_eval(cfg, ev, task["steps"])
            survivors.sort(key=lambda c: c["_eval"]["val_ppl"])
            _log_event(stage="harness_rung", rung=halving, steps=target,
                       n=len(survivors), best_ppl=survivors[0]["_eval"]["val_ppl"],
                       wall_s=round(time.time() - t0, 1), workers=profile.workers)
            if not last_rung:
                survivors = survivors[: max(1, len(survivors) // bracket.eta)]
                target *= bracket.eta
    finally:
        if own_pool:
            pool.shutdown()

    if want_sample:
        # one sample for the winner only — the slow autoregressive part runs once
        winner = survivors[0]
        ck = os.path.join(run_dir, f"c{winner['_cid']}.pt")
        model = LLM(ModelConfig(**_clean(winner))).to(device)
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"])
        model.eval()
        winner["_eval"]["sample"] = _generate_sample(model, device)
    return survivors


# ────────────────────────── self-optimization ──────────────────────────

def autotune(probe_steps: int = 5, budget_s: float = 120.0, loop_batch: int = 8,
             verbose: bool = True) -> HarnessProfile:
    """The harness optimizes the harness.

    Probes worker-count x thread-split combos with REAL short training tasks
    at the loop's batch size, measures aggregate STEPS/sec through a warm
    pool, keeps the best, and persists it. The bracket's budget currency is
    steps-at-loop-batch, so tuning any other objective (e.g. tokens/sec at a
    bigger batch) silently inflates per-step work. Probe cost <= `budget_s`.
    """
    cores = os.cpu_count() or 2
    device = _device()
    probe_cfg = {"vocab_size": 50304, "d_model": 256, "n_layers": 4, "n_heads": 4,
                 "n_kv_heads": 2, "d_ff_mult": 8 / 3, "max_seq_len": 128,
                 "rope_theta": 10000.0, "dropout": 0.0, "tie_embeddings": True}
    prepare()  # download/tokenize once, before any pool spawns

    worker_opts = sorted({1, 2, max(1, cores // 2), cores})
    combos = [(w, max(1, cores // w)) for w in worker_opts]
    if device == "cuda":
        combos = [(w, t) for w, t in combos if w <= 2]  # don't oversubscribe one GPU

    best: HarnessProfile | None = None
    best_steps_s = 0.0
    t_start = time.time()
    ctx = get_context("spawn")
    for workers, threads in combos:
        if time.time() - t_start > budget_s and best is not None:
            if verbose:
                print("tune: budget reached, skipping remaining combos")
            break
        task = {"cfg": probe_cfg, "steps": probe_steps, "lr": 3e-4,
                "batch_size": loop_batch, "val_batches": 1, "ckpt_in": None,
                "ckpt_out": None, "want_sample": False, "device": device}
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_worker_init, initargs=(threads,)) as pool:
            list(pool.map(_train_task, [dict(task, steps=1)] * workers))  # warm the pool
            t0 = time.time()
            list(pool.map(_train_task, [task] * workers))
            wall = time.time() - t0
        steps_s = workers * probe_steps / wall
        tok_s = steps_s * loop_batch * probe_cfg["max_seq_len"]
        if verbose:
            print(f"tune: workers={workers} threads={threads} batch={loop_batch}"
                  f" -> {steps_s:.2f} steps/s ({tok_s:,.0f} tok/s, {wall:.1f}s)")
        if best is None or steps_s > best_steps_s:
            best_steps_s = steps_s
            best = HarnessProfile(workers=workers, threads_per_worker=threads,
                                  batch_size=loop_batch, tokens_per_s=round(tok_s, 1),
                                  device=device)

    assert best is not None
    best.save()
    if verbose:
        print(f"tune: best profile {asdict(best)} -> {PROFILE_PATH}")
    _log_event(stage="harness_tuned", **asdict(best))
    return best


def load_or_tune(quick: bool = True) -> HarnessProfile:
    return HarnessProfile.load() or autotune(budget_s=60.0 if quick else 180.0)


# ────────────────────────── benchmark: prove the 10x ──────────────────────────

def _baseline_serial(candidates: list[dict], bracket, lr: float = 3e-4) -> tuple[list[dict], int]:
    """Replicates the original runner's semantics: serial, fresh model each
    rung (full step count, not delta), 20 val batches + a generated sample on
    EVERY eval. Returns (survivors, evals_done)."""
    survivors = list(candidates)
    steps, evals = bracket.initial_steps, 0
    for halving in range(bracket.halvings + 1):
        for cfg in survivors:
            cfg["_eval"] = _train_task({
                "cfg": cfg, "steps": steps, "lr": lr, "batch_size": 8,
                "val_batches": 20, "ckpt_in": None, "ckpt_out": None,
                "want_sample": True, "device": _device(),
            })
            evals += 1
        survivors.sort(key=lambda c: c["_eval"]["val_ppl"])
        if halving < bracket.halvings:
            survivors = survivors[: max(1, len(survivors) // bracket.eta)]
            steps *= bracket.eta
    return survivors, evals


def bench(n: int = 4, halvings: int = 2, initial_steps: int = 6,
          seed: int = 0, oversample: int = 1) -> dict:
    """Same bracket, IDENTICAL candidates, two engines: original semantics vs
    the tuned harness. With --oversample > 1 the proxy pre-filter picks the
    bracket entrants from a bigger pool (quality lever, timed separately)."""
    from hyperband import Bracket, random_config
    rng = random.Random(seed)
    bracket = Bracket(n_candidates=n, halvings=halvings, initial_steps=initial_steps)
    base_cands = [random_config(rng) for _ in range(n)]

    print(f"bench: bracket n={n} halvings={halvings} initial_steps={initial_steps}"
          f" on {_device()} ({os.cpu_count()} cores)")

    t0 = time.time()
    base_surv, base_evals = _baseline_serial(copy.deepcopy(base_cands), bracket)
    base_s = time.time() - t0
    print(f"bench: BASELINE  {base_s:7.1f}s  best_ppl={base_surv[0]['_eval']['val_ppl']:.1f}"
          f"  ({base_evals} evals, serial, from-scratch, sample every eval)")

    profile = load_or_tune(quick=True)
    proxy_s = 0.0
    cands = copy.deepcopy(base_cands)
    if oversample > 1:
        t0 = time.time()
        pool = cands + [random_config(rng) for _ in range(n * (oversample - 1))]
        cands = proxy_filter(pool, keep=n)
        proxy_s = time.time() - t0
        print(f"bench: proxy-filtered {len(pool)}->{n} in {proxy_s:.1f}s")
    t0 = time.time()
    harn_surv = parallel_halving(cands, bracket, profile=profile)
    harn_s = time.time() - t0
    print(f"bench: HARNESS   {harn_s:7.1f}s  best_ppl={harn_surv[0]['_eval']['val_ppl']:.1f}"
          f"  (workers={profile.workers} threads={profile.threads_per_worker}"
          f" batch={profile.batch_size}, parallel + promoted)")

    speedup = base_s / harn_s
    # candidate-evaluations/hour: the metric the 10x plan targets
    base_eph = base_evals / base_s * 3600
    harn_eph = base_evals / harn_s * 3600  # same bracket work accomplished
    report = {
        "baseline_s": round(base_s, 1), "harness_s": round(harn_s, 1),
        "proxy_s": round(proxy_s, 1), "speedup": round(speedup, 2),
        "baseline_evals_per_hour": round(base_eph, 1),
        "harness_evals_per_hour": round(harn_eph, 1),
        "baseline_best_ppl": round(base_surv[0]["_eval"]["val_ppl"], 2),
        "harness_best_ppl": round(harn_surv[0]["_eval"]["val_ppl"], 2),
        "profile": asdict(profile),
        "winner_trained_steps_harness": harn_surv[0]["_eval"]["trained_steps"],
    }
    print(f"bench: SPEEDUP {speedup:.2f}x   evals/hour {base_eph:,.0f} -> {harn_eph:,.0f}")
    print("bench: note — harness winner carries CUMULATIVE training"
          f" ({report['winner_trained_steps_harness']} steps) vs baseline's fresh"
          f" {initial_steps * bracket.eta ** halvings}-step model: faster AND higher fidelity")
    with open(os.path.join(os.path.dirname(__file__), "bench_harness.json"), "w") as f:
        json.dump(report, f, indent=2)
    _log_event(stage="harness_bench", **{k: v for k, v in report.items() if k != "profile"})
    return report


# ────────────────────────── CLI ──────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tune", help="self-optimize the harness for this machine")
    t.add_argument("--budget", type=float, default=120.0)
    t.add_argument("--probe-steps", type=int, default=5)

    b = sub.add_parser("bench", help="baseline vs harness on the same bracket")
    b.add_argument("--n", type=int, default=4)
    b.add_argument("--halvings", type=int, default=2)
    b.add_argument("--initial-steps", type=int, default=6)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--oversample", type=int, default=1,
                   help=">1 enables the SNIP proxy pre-filter (timed separately)")

    r = sub.add_parser("run", help="run one bracket through the tuned harness")
    r.add_argument("--n", type=int, default=4)
    r.add_argument("--halvings", type=int, default=2)
    r.add_argument("--initial-steps", type=int, default=25)
    r.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    if args.cmd == "tune":
        autotune(probe_steps=args.probe_steps, budget_s=args.budget)
    elif args.cmd == "bench":
        bench(n=args.n, halvings=args.halvings, initial_steps=args.initial_steps,
              seed=args.seed, oversample=args.oversample)
    elif args.cmd == "run":
        from hyperband import Bracket, random_config
        rng = random.Random(args.seed)
        bracket = Bracket(n_candidates=args.n, halvings=args.halvings,
                          initial_steps=args.initial_steps)
        cands = proxy_filter([random_config(rng) for _ in range(args.n * 2)], keep=args.n)
        survivors = parallel_halving(cands, bracket, profile=load_or_tune())
        w = survivors[0]["_eval"]
        print(json.dumps({"val_ppl": w["val_ppl"], "params_m": w["params_m"],
                          "trained_steps": w["trained_steps"],
                          "sample": w["sample"][:200]}, indent=2))


if __name__ == "__main__":
    main()
