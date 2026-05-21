"""Benchmarks: perplexity, throughput, memory, and a tiny LAMBADA-style completion eval.

Output → bench.json, consumed by the web dashboard.
"""
from __future__ import annotations

import json
import math
import os
import time

import torch

from data import Loader, prepare, tokenizer
from model import LLM, ModelConfig
from train import CKPT_PATH, pick_device

OUT = os.path.join(os.path.dirname(__file__), "bench.json")

# Micro-eval: given a prompt, does the model rank the correct completion
# higher than distractors? (Shakespeare-flavored to match training data.)
COMPLETIONS = [
    {"prompt": "To be, or not to be, that is the ",
     "correct": "question", "distractors": ["answer", "problem", "apple"]},
    {"prompt": "All the world's a ",
     "correct": "stage", "distractors": ["store", "stone", "stag"]},
    {"prompt": "Romeo, Romeo, wherefore art thou ",
     "correct": "Romeo", "distractors": ["Hamlet", "Caesar", "Brutus"]},
    {"prompt": "What light through yonder window ",
     "correct": "breaks", "distractors": ["shines", "falls", "stops"]},
    {"prompt": "Friends, Romans, countrymen, lend me your ",
     "correct": "ears", "distractors": ["eyes", "swords", "gold"]},
]


def load_model(device: str):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["cfg"])
    model = LLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def score_sequence(model, enc, device, prompt: str, completion: str) -> float:
    """Sum log-prob of `completion` tokens conditioned on `prompt`."""
    prompt_ids = enc.encode_ordinary(prompt)
    comp_ids = enc.encode_ordinary(completion)
    ids = torch.tensor([prompt_ids + comp_ids], dtype=torch.long, device=device)
    logits, _ = model(ids)
    logprobs = torch.log_softmax(logits[0], dim=-1)
    total = 0.0
    for i, t in enumerate(comp_ids):
        total += logprobs[len(prompt_ids) + i - 1, t].item()
    return total / max(len(comp_ids), 1)


def completion_accuracy(model, enc, device) -> tuple[float, list[dict]]:
    correct = 0
    details = []
    for item in COMPLETIONS:
        options = [item["correct"]] + item["distractors"]
        scores = [score_sequence(model, enc, device, item["prompt"], o) for o in options]
        picked = options[int(torch.tensor(scores).argmax())]
        ok = picked == item["correct"]
        correct += int(ok)
        details.append({"prompt": item["prompt"], "correct": item["correct"],
                        "picked": picked, "scores": dict(zip(options, scores))})
    return correct / len(COMPLETIONS), details


@torch.no_grad()
def throughput(model, cfg: ModelConfig, device: str) -> dict:
    model.eval()
    bsz, T = 4, cfg.max_seq_len
    x = torch.randint(0, cfg.vocab_size, (bsz, T), device=device)
    # warmup
    for _ in range(3):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    iters = 20
    for _ in range(iters):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    dt = time.time() - t0
    toks = bsz * T * iters
    return {"forward_tok_per_sec": toks / dt, "batch": bsz, "seq_len": T}


def main():
    device, _ = pick_device()
    print(f"Benchmarking on {device}")

    if not os.path.exists(CKPT_PATH):
        raise SystemExit("No checkpoint found — run `python3 train.py` first.")

    model, cfg = load_model(device)
    enc = tokenizer()

    # 1. perplexity on val split
    _, val_bin = prepare()
    val_loader = Loader(val_bin, cfg.max_seq_len, 16, device)
    losses = []
    with torch.no_grad():
        for _ in range(50):
            x, y = val_loader.batch()
            _, loss = model(x, y)
            losses.append(loss.item())
    val_loss = sum(losses) / len(losses)
    ppl = math.exp(val_loss)
    print(f"  val_loss={val_loss:.4f}  perplexity={ppl:.2f}")

    # 2. completion accuracy
    acc, detail = completion_accuracy(model, enc, device)
    print(f"  completion accuracy: {acc * 100:.1f}% ({sum(1 for d in detail if d['picked'] == d['correct'])}/{len(detail)})")

    # 3. throughput
    tp = throughput(model, cfg, device)
    print(f"  forward throughput: {tp['forward_tok_per_sec']:,.0f} tok/s")

    # 4. memory footprint
    n_params = model.num_params()
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())

    # 5. sample generation
    prompt = "ROMEO:\n"
    ids = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=80, temperature=0.8, top_k=40)
    sample = enc.decode(out[0].tolist())

    result = {
        "device": device,
        "params_m": round(n_params / 1e6, 2),
        "weights_mb": round(bytes_ / 1e6, 2),
        "val_loss": val_loss,
        "perplexity": ppl,
        "completion_accuracy": acc,
        "completion_detail": detail,
        "throughput": tp,
        "sample_prompt": prompt,
        "sample_output": sample,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {OUT}")
    print("\n--- Sample ---")
    print(sample)


if __name__ == "__main__":
    main()
