"""Prior-guided evolutionary search — the breeding arm (10X_PLAN Pillar 1).

Every other arm *samples* the config space (random draws it uniformly;
hyperband/prior draw a pool and filter). None of them *breed*: recombine the
architectures that actually won into new ones. This arm does, and it fuses the
project's two existing pillars into the genetic operators:

  genome    a hyperparameter config (hyperband.random_config schema):
            d_model, n_layers, n_heads, n_kv_heads, d_ff_mult, tie_embeddings
  fitness   deterministic val_ppl from the throughput harness (_train_task),
            lower is better — the SAME eval every arm is scored by
  selection truncation + elitism: the best `elite` genomes seed the next
            generation and carry their fitness forward without re-training
  crossover uniform gene mix of two elite parents, then repaired to a valid
            architecture (n_heads must divide d_model, n_kv_heads <= n_heads)
  mutation  local moves along each gene's ordered domain (a d_model step up or
            down, +/- a layer) — neighbourhood search, not a random restart

The novel lever is PRIOR-GUIDED BREEDING. For every offspring slot we generate
`oversample` candidate children and score them with the CheapPrior surrogate —
the same cheap (config -> ppl) predictor the `prior` arm uses for acquisition —
then spend real training steps only on the child the surrogate likes best. The
prior is refit from every real eval, so as generations accrue the screening
sharpens. Evolution proposes; the surrogate pre-screens; the harness verifies.

The only difference from the `random` arm is where the next genome comes from
(bred + screened vs drawn uniformly). Same budget, same eval, same accounting,
so `steps-to-random-target` reads as a clean "does breeding beat sampling?".

Every evaluated genome is recorded with its id, generation, parents and origin
so `evolve_viz.py` can draw the genealogy. Run standalone:

    python3 evolve.py run --quick        # 1 seed, small budget (~4 min CPU)

writes evolve_report.json (+ lineage) consumed by the visualization.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from harness import (
    HarnessProfile, load_or_tune, _train_task, _worker_init, _device, RUNS_DIR,
)
from hyperband import CheapPrior, random_config

VAL_BATCHES = 8   # deterministic windows — identical to arms.py so ppl is comparable
REPORT_JSON = os.path.join(os.path.dirname(__file__), "evolve_report.json")

# Ordered gene domains. Order matters: mutation is a step to an ADJACENT value,
# so the list order defines the neighbourhood (small -> large capacity).
D_MODELS = [192, 256, 320, 384, 448, 512]
N_LAYERS = [2, 4, 6, 8]
D_FF_MULTS = [2.0, 8 / 3, 3.0, 4.0]
GENES = ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff_mult", "tie_embeddings")


# ────────────────────────── genome validity ──────────────────────────

def _valid_heads(d_model: int) -> list[int]:
    """Head counts that evenly divide d_model (a hard model constraint)."""
    return [h for h in (2, 4, 6, 8) if d_model % h == 0]


def repair(cfg: dict) -> dict:
    """Make a (possibly crossed-over) genome a buildable architecture.

    Crossover can pair a d_model from one parent with an n_heads from another
    that doesn't divide it, or an n_kv_heads that exceeds n_heads. Repair snaps
    each violated gene to the nearest valid value instead of rejecting the
    child, so recombination stays productive.
    """
    cfg = dict(cfg)
    valid = _valid_heads(cfg["d_model"])
    if cfg["n_heads"] not in valid:
        cfg["n_heads"] = min(valid, key=lambda h: abs(h - cfg["n_heads"]))
    # grouped-query attention requires n_kv_heads to DIVIDE n_heads (not merely
    # be <=). Every element of {1, 2, n_heads} divides n_heads for the even head
    # counts we use; snap an out-of-set gene to the nearest legal divisor.
    allowed_kv = [k for k in sorted({1, 2, cfg["n_heads"]}) if cfg["n_heads"] % k == 0]
    if cfg["n_kv_heads"] not in allowed_kv:
        cfg["n_kv_heads"] = min(allowed_kv, key=lambda k: abs(k - cfg["n_kv_heads"]))
    return cfg


def _full(cfg: dict) -> dict:
    """Attach the fixed (non-evolved) fields a ModelConfig needs."""
    return {"vocab_size": 50304, "max_seq_len": 128, "rope_theta": 10000.0,
            "dropout": 0.0, **{g: cfg[g] for g in GENES}}


# ────────────────────────── genetic operators ──────────────────────────

def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    """Uniform crossover: each gene comes from parent a or b with equal odds."""
    child = {g: (a if rng.random() < 0.5 else b)[g] for g in GENES}
    return repair(child)


def _step(value, domain: list, rng: random.Random):
    """Move `value` one position along an ordered domain (clamped at the ends)."""
    i = domain.index(value) if value in domain else \
        min(range(len(domain)), key=lambda j: abs(domain[j] - value))
    i = max(0, min(len(domain) - 1, i + rng.choice((-1, 1))))
    return domain[i]


def mutate(cfg: dict, rng: random.Random, rate: float = 0.34) -> dict:
    """Perturb each gene independently with probability `rate`, locally.

    Ordered genes take one step to an adjacent value (neighbourhood search);
    head counts jump to another valid divisor of the current d_model.
    """
    cfg = dict(cfg)
    if rng.random() < rate:
        cfg["d_model"] = _step(cfg["d_model"], D_MODELS, rng)
    if rng.random() < rate:
        cfg["n_layers"] = _step(cfg["n_layers"], N_LAYERS, rng)
    if rng.random() < rate:
        cfg["d_ff_mult"] = _step(cfg["d_ff_mult"], D_FF_MULTS, rng)
    if rng.random() < rate:
        cfg["n_heads"] = rng.choice(_valid_heads(cfg["d_model"]))
    if rng.random() < rate:
        cfg["n_kv_heads"] = rng.choice(sorted({1, 2, cfg["n_heads"]}))
    return repair(cfg)


def _key(cfg: dict) -> tuple:
    """Identity of a genome for de-duplication (genes only)."""
    return tuple(cfg[g] for g in GENES)


# ────────────────────────── the evolutionary loop ──────────────────────────

class Genome:
    __slots__ = ("id", "gen", "cfg", "parents", "origin", "ppl", "params_m")

    def __init__(self, gid, gen, cfg, parents, origin):
        self.id, self.gen, self.cfg = gid, gen, dict(cfg)
        self.parents, self.origin = parents, origin
        self.ppl, self.params_m = math.inf, 0.0

    def node(self) -> dict:
        return {"id": self.id, "gen": self.gen, "parents": self.parents,
                "origin": self.origin, "ppl": round(self.ppl, 3),
                "params_m": self.params_m, "cfg": self.cfg}


def _breed(parents: list[Genome], prior: CheapPrior, rng: random.Random,
           taken: set, oversample: int) -> tuple[dict, list[str]]:
    """Prior-guided offspring: draw `oversample` candidate children from two
    random elite parents, score each with the surrogate, keep the best unseen
    one. Falls back to whatever's available if every candidate collides."""
    best, best_score, best_par = None, math.inf, []
    for _ in range(oversample):
        pa, pb = rng.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
        child = mutate(crossover(pa.cfg, pb.cfg, rng), rng)
        score = prior.acquisition(child) if prior.X else rng.random()
        if _key(child) in taken:
            score += 10.0   # heavily penalise duplicates, don't hard-ban them
        if score < best_score:
            best, best_score, best_par = child, score, [pa.id, pb.id]
    return best, best_par


def run_evolve(seed: int, budget: int, full_steps: int, profile: HarnessProfile,
               pool: ProcessPoolExecutor, pop: int = 6, elite: int = 3,
               oversample: int = 6, seed_prior: CheapPrior | None = None
               ) -> dict:
    """Evolve architectures under a fixed training-step budget.

    Returns {"trajectory": [...], "lineage": [...], "best": {...}} — the
    trajectory has the same shape arms.py records (best-ppl-so-far vs steps
    spent), so the evolutionary arm is directly comparable to random/hyperband.
    """
    rng = random.Random(seed)
    prior = CheapPrior() if seed_prior is None else seed_prior
    lineage: list[Genome] = []
    counter = 0

    def new(cfg, gen, parents, origin) -> Genome:
        nonlocal counter
        g = Genome(f"g{counter}", gen, cfg, parents, origin)
        counter += 1
        return g

    def evaluate(genomes: list[Genome]) -> None:
        """Train every genome to full depth through the harness, in parallel."""
        tasks = [{
            "cfg": _full(g.cfg), "steps": full_steps, "lr": 3e-4,
            "batch_size": profile.batch_size, "val_batches": VAL_BATCHES,
            "ckpt_in": None, "ckpt_out": None, "want_sample": False,
            "device": _device(), "deterministic_val": True,
        } for g in genomes]
        for g, ev in zip(genomes, pool.map(_train_task, tasks)):
            g.ppl, g.params_m = ev["val_ppl"], ev["params_m"]
            prior.add(g.cfg, ev["val_ppl"], steps=ev["trained_steps"])

    # trajectory bookkeeping — identical accounting to arms.Trajectory
    traj, spent, best = [], 0, math.inf

    def record(genomes: list[Genome]) -> None:
        nonlocal spent, best
        for g in genomes:
            spent += full_steps
            best = min(best, g.ppl)
            traj.append({"steps": spent, "ppl": round(g.ppl, 3),
                         "best": round(best, 3), "gen": g.gen})

    # ── generation 0: a fresh random population (prior-seeded if warm) ──
    gen0: list[Genome] = []
    taken: set = set()

    def _draw() -> dict:
        # take ALL genes from ONE random_config so n_heads/n_kv stay a mutually
        # valid architecture (a per-gene draw mixes incompatible heads/kv)
        base = random_config(rng)
        return repair({g: base[g] for g in GENES})

    while len(gen0) < pop:
        cfg = _draw()
        if prior.X:   # warm prior: bias the very first draw toward good regions
            cfg = min((_draw() for _ in range(oversample)), key=prior.acquisition)
        if _key(cfg) in taken:
            continue
        taken.add(_key(cfg))
        gen0.append(new(cfg, 0, [], "random"))
    evaluate(gen0)
    record(gen0)
    lineage.extend(gen0)
    population = gen0

    # ── subsequent generations: elitism + prior-guided breeding ──
    gen = 1
    while budget - spent >= (pop - elite) * full_steps:
        ranked = sorted(population, key=lambda g: g.ppl)
        parents = ranked[:elite]                 # truncation selection
        taken = {_key(p.cfg) for p in parents}   # elites already occupy the gen
        children: list[Genome] = []
        for _ in range(pop - elite):             # elites carried over, not re-trained
            cfg, par = _breed(parents, prior, rng, taken, oversample)
            taken.add(_key(cfg))
            children.append(new(cfg, gen, par, "crossover"))
        evaluate(children)
        record(children)
        lineage.extend(children)
        population = parents + children          # elites persist by fitness
        gen += 1

    ranked = sorted(lineage, key=lambda g: g.ppl)
    return {"trajectory": traj, "lineage": [g.node() for g in lineage],
            "best": ranked[0].node(), "generations": gen, "spent": spent,
            "pop": pop, "elite": elite, "oversample": oversample}


# ────────────────────────── driver ──────────────────────────

def run(seeds: int, budget: int, full_steps: int, pop: int, elite: int,
        oversample: int, warm_prior_path: str | None = None) -> dict:
    profile = load_or_tune(quick=True)
    from data import prepare
    prepare()   # once, before the pool spawns workers

    seed_prior = CheapPrior.load(warm_prior_path) if warm_prior_path else None
    runs = []
    with ProcessPoolExecutor(max_workers=profile.workers,
                             mp_context=get_context("spawn"),
                             initializer=_worker_init,
                             initargs=(profile.threads_per_worker,)) as pool:
        for seed in range(seeds):
            t0 = time.time()
            # each seed gets its own prior copy so seeds stay independent
            sp = CheapPrior.load(warm_prior_path) if warm_prior_path else None
            res = run_evolve(seed, budget, full_steps, profile, pool,
                             pop=pop, elite=elite, oversample=oversample,
                             seed_prior=sp)
            res["seed"] = seed
            runs.append(res)
            b = res["best"]
            print(f"evolve: seed={seed} best_ppl={b['ppl']:8.2f} "
                  f"({b['cfg']['d_model']}d x{b['cfg']['n_layers']}L, "
                  f"{b['params_m']}M) gens={res['generations']} "
                  f"spent={res['spent']} evals={len(res['lineage'])} "
                  f"({time.time() - t0:.0f}s)")

    report = {"budget_steps": budget, "full_steps": full_steps,
              "val_batches": VAL_BATCHES, "deterministic_val": True,
              "pop": pop, "elite": elite, "oversample": oversample,
              "runs": runs}
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"evolve: wrote {REPORT_JSON} "
          f"({sum(len(r['lineage']) for r in runs)} genomes across {seeds} seed(s))")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evolve architectures at a fixed step budget")
    r.add_argument("--seeds", type=int, default=1)
    r.add_argument("--budget", type=int, default=192, help="training steps per seed")
    r.add_argument("--pop", type=int, default=6, help="population size per generation")
    r.add_argument("--elite", type=int, default=3, help="survivors carried + bred from")
    r.add_argument("--oversample", type=int, default=6,
                   help="candidate children scored by the prior per offspring slot")
    r.add_argument("--full-steps", type=int, default=None,
                   help="training steps per genome eval (default: budget//24*4); "
                        "lower it to fit more generations into the same budget")
    r.add_argument("--quick", action="store_true", help="1 seed x 96 steps (~4 min CPU)")
    r.add_argument("--warm-prior", metavar="PATH", default=None,
                   help="seed generation 0 and screening from an accumulated prior")
    args = ap.parse_args()

    if args.cmd == "run":
        seeds, budget = args.seeds, args.budget
        if args.quick:
            seeds, budget = 1, 96
        # full_steps mirrors arms.py's bracket final depth so ppl scales match;
        # override to trade eval depth for more generations at a fixed budget
        full_steps = args.full_steps or max(2, budget // 24) * 4
        run(seeds=seeds, budget=budget, full_steps=full_steps, pop=args.pop,
            elite=args.elite, oversample=args.oversample,
            warm_prior_path=args.warm_prior)


if __name__ == "__main__":
    main()
