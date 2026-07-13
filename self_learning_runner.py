"""Main self-learning loop. Glues agents + hyperband + braintrust + dashboard.

Run with:
    python3 self_learning_runner.py --max-steps 100 --rounds 4

The dashboard (server_v3.py + dashboard.html) reads the snapshot file this writes.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from typing import Any

import torch

from agents import OrchestratorAgent
from braintrust_bridge import (
    log_event, reset_state, write_snapshot, fetch_recent_logs, read_human_queue,
)
from hyperband import (
    CheapPrior, load_prior_from_experiments, random_config,
    successive_halving, standard_brackets, Bracket,
)

from data import prepare, Loader  # existing repo modules
from model import LLM, ModelConfig


# ──────────────────────────── partial training ────────────────────────────

def train_partial(cfg: dict, steps: int, lr: float = 3e-4) -> dict:
    """Train a fresh model for `steps` mini-batches and return eval dict."""
    from harness import _clean  # strips _eval/_steps added by earlier rungs
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    mc = ModelConfig(**_clean(cfg))
    model = LLM(mc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    train_bin, val_bin = prepare()
    train_loader = Loader(train_bin, block_size=mc.max_seq_len, batch_size=8, device=device)
    val_loader = Loader(val_bin, block_size=mc.max_seq_len, batch_size=8, device=device)
    model.train()
    t0 = time.time()
    losses = []
    for i in range(steps):
        x, y = train_loader.batch()
        _, loss = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    # eval
    model.eval()
    val_losses = []
    with torch.no_grad():
        for j in range(20):
            x, y = val_loader.batch()
            _, loss = model(x, targets=y)
            val_losses.append(loss.item())
    val_loss = float(sum(val_losses) / max(len(val_losses), 1))
    val_ppl = float(2.71828 ** val_loss)

    # quick sample
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    prompt = "ROMEO:\n"
    ids = torch.tensor([enc.encode(prompt)], device=device)
    out = model.generate(ids, max_new_tokens=80, temperature=0.8, top_k=50)
    sample = enc.decode(out[0].tolist())

    return {
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "cloze_accuracy": 0.0,  # cheap eval — leave for full benchmark
        "tokens_seen": steps * 8 * mc.max_seq_len,
        "elapsed_s": round(time.time() - t0, 1),
        "loss_curve": losses,
        "sample": sample,
        "params_m": round(model.num_params() / 1e6, 2),
    }


def trainer_callable_for_agents(cfg: dict, lr: float) -> dict:
    """Adapter so OrchestratorAgent can call train_partial via train_fn."""
    res = train_partial(cfg, steps=100, lr=lr)
    return {
        "raw_metrics": {k: v for k, v in res.items() if k != "sample"},
        "sample": res["sample"],
    }


# ──────────────────────────── main loop ────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3, help="how many self-learning rounds")
    ap.add_argument("--max-steps", type=int, default=100, help="steps for top survivor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reset", action="store_true", help="clear local state first")
    ap.add_argument("--harness", action="store_true",
                    help="use the tuned throughput harness (parallel workers + "
                         "checkpoint promotion); auto-tunes on first use")
    args = ap.parse_args()

    harness_profile = None
    if args.harness:
        from harness import load_or_tune
        harness_profile = load_or_tune(quick=True)

    rng = random.Random(args.seed)
    if args.reset:
        reset_state()

    prior = load_prior_from_experiments("experiments.json")
    prior.extend(CheapPrior.load("prior_store.json"))   # compound across runs
    orchestrator = OrchestratorAgent()
    history: list[dict] = []
    bracket = Bracket(n_candidates=4, halvings=2, initial_steps=max(1, args.max_steps // 4))

    snapshot: dict[str, Any] = {
        "started_at": time.time(),
        "rounds": [],
        "prior_size": len(prior.X),
        "bracket": {"n": bracket.n_candidates, "halvings": bracket.halvings, "init_steps": bracket.initial_steps},
        "current": {"round": 0, "phase": "init"},
    }
    write_snapshot(snapshot)

    for r in range(args.rounds):
        snapshot["current"] = {"round": r + 1, "phase": "proposing"}
        write_snapshot(snapshot)

        # ── propose candidates: 1 from agent + 3 prior-seeded random + sorted by acquisition
        candidates = []
        try:
            ag_prop = orchestrator.trainer.propose([
                {k: v for k, v in h.items() if k in ("name", "config", "val_ppl", "params_m")}
                for h in history[-10:]
            ])
            candidates.append(asdict(ag_prop)["config"])
        except Exception as e:
            log_event(stage="trainer_fallback", error=str(e))

        for _ in range(bracket.n_candidates - len(candidates)):
            cands = [random_config(rng) for _ in range(20)]
            cands.sort(key=prior.acquisition)
            candidates.append(cands[0])

        # ── hyperband bracket
        snapshot["current"] = {"round": r + 1, "phase": "training"}
        write_snapshot(snapshot)
        if harness_profile is not None:
            from harness import parallel_halving
            survivors = parallel_halving(candidates, bracket, profile=harness_profile)
        else:
            survivors = successive_halving(candidates, train_partial, bracket)
        winner = survivors[0]

        # ── multi-agent eval/judge/meta on the winner
        snapshot["current"] = {"round": r + 1, "phase": "evaluating"}
        write_snapshot(snapshot)
        ev = winner["_eval"]
        agent_step = orchestrator.step(
            history=[{"name": h.get("name", f"r{i}"), "config": h["config"],
                      "val_ppl": h["val_ppl"], "params_m": h.get("params_m", 0)}
                     for i, h in enumerate(history)],
            train_fn=lambda cfg, lr: {
                "raw_metrics": {k: v for k, v in ev.items() if k != "sample"},
                "sample": ev["sample"],
            },
        )

        # ── update prior with all candidates (not just winner) — more learning per round
        for c in survivors:
            prior.add(c, c["_eval"]["val_ppl"])
        prior.save("prior_store.json")   # future runs start from this knowledge
        history.append({
            "name": f"round{r+1}-winner",
            "config": winner,
            "val_ppl": winner["_eval"]["val_ppl"],
            "params_m": winner["_eval"]["params_m"],
            "agent_step": agent_step,
        })

        snapshot["rounds"].append({
            "round": r + 1,
            "candidates": len(candidates),
            "survivors": len(survivors),
            "winner_ppl": winner["_eval"]["val_ppl"],
            "winner_params_m": winner["_eval"]["params_m"],
            "agent_step": agent_step,
            "prior_size": len(prior.X),
        })
        snapshot["current"] = {"round": r + 1, "phase": "done"}
        snapshot["braintrust_traces"] = fetch_recent_logs(limit=30)
        snapshot["human_queue"] = read_human_queue()
        write_snapshot(snapshot)
        log_event(stage="round_done", round=r + 1, winner_ppl=winner["_eval"]["val_ppl"])

    snapshot["finished_at"] = time.time()
    snapshot["current"] = {"round": args.rounds, "phase": "complete"}
    snapshot["braintrust_traces"] = fetch_recent_logs(limit=50)
    write_snapshot(snapshot)
    print(json.dumps(snapshot["rounds"], indent=2, default=str))


if __name__ == "__main__":
    main()
