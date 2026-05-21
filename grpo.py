"""GRPO (Group Relative Policy Optimization) — DeepSeek-R1's RL algorithm.

Why GRPO over PPO:
  - No value (critic) network — saves ~50% memory + params
  - No advantage model — uses group-relative reward as advantage
  - Simpler loss, fewer hyperparameters
  - Empirically matches PPO on reasoning tasks

Algorithm (per step):
  1. Sample G completions per prompt from current policy π_θ
  2. Score each with reward function R (rule-based or judge-LLM)
  3. Group-normalize: A_i = (R_i - mean(R)) / std(R)
  4. Compute KL to reference policy π_ref (frozen snapshot)
  5. Loss: -E[ A_i * log π_θ(y_i|x) ] + β * KL(π_θ || π_ref)

For tiny LLM on Tiny Shakespeare we use:
  - Reward: cloze accuracy (5 hand-picked Shakespeare lines)
  - G = 4 completions per prompt
  - β (KL coef) = 0.04 (DeepSeek default)
  - Reference = frozen copy of the SFT-tuned model
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from model import LLM


@dataclass
class GRPOConfig:
    group_size: int = 4              # G — completions per prompt
    kl_coef: float = 0.04            # β
    lr: float = 1e-5                 # GRPO needs lower LR than SFT
    max_new_tokens: int = 64
    temperature: float = 0.9
    clip_eps: float = 0.2            # PPO-style clipping (optional)
    grad_clip: float = 1.0


# Cloze prompts → (prompt, target_tok_id) — minimal Shakespeare reward
CLOZE_PROBES = [
    ("To be, or not to be, that is the ", "question"),
    ("All the world's a ", "stage"),
    ("Romeo, Romeo, wherefore art thou ", "Romeo"),
    ("What light through yonder window ", "breaks"),
    ("Friends, Romans, countrymen, lend me your ", "ears"),
]


def cloze_reward(model: LLM, enc, device: str) -> tuple[float, list[bool]]:
    """Per-prompt 0/1 reward — does the model rank the correct token #1?"""
    model.eval()
    hits = []
    with torch.no_grad():
        for prompt, target in CLOZE_PROBES:
            ids = torch.tensor([enc.encode(prompt)], device=device)
            logits, _ = model(ids)
            last = logits[0, -1]
            top = int(last.argmax().item())
            hits.append(top == enc.encode(" " + target.strip())[0]
                        or top == enc.encode(target)[0])
    return sum(hits) / len(hits), hits


def sample_completions(model: LLM, prompt_ids: torch.Tensor, cfg: GRPOConfig,
                       enc) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample G completions. Returns (completion_ids [G,T], log_probs [G,T])."""
    G = cfg.group_size
    model.eval()
    completions = []
    log_probs_per_completion = []

    for _ in range(G):
        ids = prompt_ids.clone()
        log_probs = []
        for _ in range(cfg.max_new_tokens):
            with torch.no_grad():
                logits, _ = model(ids[:, -model.cfg.max_seq_len:])
            logits = logits[0, -1] / cfg.temperature
            probs = F.softmax(logits, dim=-1)
            tok = torch.multinomial(probs, 1)
            log_probs.append(torch.log(probs[tok] + 1e-10))
            ids = torch.cat([ids, tok.unsqueeze(0)], dim=1)
        completions.append(ids)
        log_probs_per_completion.append(torch.stack(log_probs).squeeze())

    return completions, log_probs_per_completion


def compute_kl_div(policy_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """KL(π_θ || π_ref) on per-token distributions, summed over tokens."""
    log_p = F.log_softmax(policy_logits, dim=-1)
    log_q = F.log_softmax(ref_logits, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1).mean()


def grpo_step(
    policy: LLM,
    reference: LLM,
    optimizer: torch.optim.Optimizer,
    prompt_ids: torch.Tensor,
    reward_fn: Callable[[torch.Tensor], float],
    cfg: GRPOConfig,
    enc,
) -> dict:
    """One GRPO update: sample G, score, group-normalize, gradient step."""
    G = cfg.group_size
    completions, sampled_log_probs = sample_completions(policy, prompt_ids, cfg, enc)

    # ── reward each completion (rule-based or judge-LLM)
    rewards = torch.tensor([reward_fn(c) for c in completions], device=prompt_ids.device)

    # ── group-relative advantage (the "G" in GRPO)
    if rewards.std() > 1e-6:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    else:
        advantages = rewards - rewards.mean()  # all same → no signal

    # ── policy gradient + KL penalty
    policy.train()
    losses = []
    kls = []
    for i in range(G):
        ids = completions[i]
        logits_p, _ = policy(ids[:, :-1])
        with torch.no_grad():
            logits_r, _ = reference(ids[:, :-1])

        # PG loss on COMPLETION tokens only (not prompt)
        prompt_len = prompt_ids.size(1)
        log_p_full = F.log_softmax(logits_p, dim=-1)
        target_ids = ids[:, 1:].contiguous()
        log_p_taken = log_p_full.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        completion_log_p = log_p_taken[:, prompt_len - 1:].sum(dim=-1).mean()

        kl = compute_kl_div(logits_p[:, prompt_len - 1:], logits_r[:, prompt_len - 1:])
        kls.append(kl.item())

        loss = -(advantages[i] * completion_log_p) + cfg.kl_coef * kl
        losses.append(loss)

    total_loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
    optimizer.step()

    return {
        "loss": float(total_loss.item()),
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std().item()),
        "kl_mean": float(sum(kls) / len(kls)),
        "advantage_mean": float(advantages.mean().item()),
    }


def grpo_train(
    sft_model: LLM,
    enc,
    prompts: list[str],
    n_steps: int = 200,
    cfg: Optional[GRPOConfig] = None,
    device: str = "mps",
) -> tuple[LLM, list[dict]]:
    """Full GRPO loop. Returns (RL-tuned model, per-step metrics).

    Reference policy is a deep-copy of SFT model — frozen.
    The cloze reward is a stand-in; in practice plug in a judge-LLM reward here.
    """
    cfg = cfg or GRPOConfig()
    policy = sft_model
    reference = copy.deepcopy(sft_model).eval()
    for p in reference.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    metrics_log = []

    for step in range(n_steps):
        prompt = prompts[step % len(prompts)]
        prompt_ids = torch.tensor([enc.encode(prompt)], device=device)

        # cloze accuracy as scalar reward (per completion: did it produce coherent text?)
        # for tiny model we use a *length-of-coherent-prefix* proxy: how many of the first
        # 16 tokens are alpha (not punct/garbage)
        def reward_fn(completion_ids: torch.Tensor) -> float:
            text = enc.decode(completion_ids[0].tolist())
            tail = text[len(prompt):][:60]
            alpha_ratio = sum(c.isalpha() or c in " \n.,!?;:'-" for c in tail) / max(len(tail), 1)
            cloze_acc, _ = cloze_reward(policy, enc, device)  # global signal
            return 0.5 * alpha_ratio + 0.5 * cloze_acc

        m = grpo_step(policy, reference, opt, prompt_ids, reward_fn, cfg, enc)
        m["step"] = step
        metrics_log.append(m)

        if step % 10 == 0:
            print(f"[grpo {step:3d}] loss={m['loss']:.3f}  R={m['reward_mean']:.2f}±{m['reward_std']:.2f}  KL={m['kl_mean']:.3f}")

    return policy, metrics_log


# ────────────── SFT vs GRPO comparison harness ──────────────

def compare_sft_vs_grpo(sft_model: LLM, grpo_model: LLM, enc, device: str) -> dict:
    """Side-by-side quantitative + qualitative comparison."""
    out = {"sft": {}, "grpo": {}, "deltas": {}}

    for name, m in [("sft", sft_model), ("grpo", grpo_model)]:
        cloze, hits = cloze_reward(m, enc, device)
        m.eval()
        prompt = "ROMEO:\n"
        ids = torch.tensor([enc.encode(prompt)], device=device)
        gen = m.generate(ids, max_new_tokens=80, temperature=0.8, top_k=50)
        sample = enc.decode(gen[0].tolist())
        out[name] = {"cloze_accuracy": cloze, "cloze_hits": hits, "sample": sample,
                     "params_m": round(m.num_params() / 1e6, 2)}

    out["deltas"] = {
        "cloze_acc_delta": out["grpo"]["cloze_accuracy"] - out["sft"]["cloze_accuracy"],
    }
    return out
