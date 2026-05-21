"""Web dashboard. Single-file Flask app.

Routes:
  /                 → HTML dashboard (live training chart, benchmarks, chat, agent)
  /api/log          → streams the training JSONL log
  /api/bench        → benchmark results
  /api/generate     → POST {prompt, max_tokens, temperature} → text
  /api/agent        → POST {goal} → full trace + answer

Run locally:    python3 server.py
Expose publicly:  cloudflared tunnel --url http://localhost:8000
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading

import torch
from flask import Flask, jsonify, request, send_from_directory

from agent import Agent
from data import tokenizer
from model import LLM, ModelConfig
from train import CKPT_PATH, LOG_PATH, pick_device

ROOT = os.path.dirname(__file__)
BENCH_PATH = os.path.join(ROOT, "bench.json")
EXPERIMENTS_PATH = os.path.join(ROOT, "experiments.json")

app = Flask(__name__, static_folder=ROOT)

# Lazy singletons
_lock = threading.Lock()
_state: dict = {}


def get_model():
    with _lock:
        if "model" not in _state and os.path.exists(CKPT_PATH):
            device, _ = pick_device()
            ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
            cfg = ModelConfig(**ckpt["cfg"])
            m = LLM(cfg).to(device)
            m.load_state_dict(ckpt["model"])
            m.eval()
            _state.update(model=m, cfg=cfg, device=device, enc=tokenizer())
        return _state.get("model"), _state.get("cfg"), _state.get("device"), _state.get("enc")


def get_agent():
    with _lock:
        if "agent" not in _state:
            _state["agent"] = Agent()
        return _state["agent"]


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/api/log")
def api_log():
    if not os.path.exists(LOG_PATH):
        return jsonify(events=[])
    events = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return jsonify(events=events)


@app.get("/api/bench")
def api_bench():
    if not os.path.exists(BENCH_PATH):
        return jsonify(ready=False)
    with open(BENCH_PATH) as f:
        return jsonify(ready=True, **json.load(f))


_train_proc: subprocess.Popen | None = None


@app.post("/api/train/start")
def api_train_start():
    global _train_proc
    with _lock:
        if _train_proc is not None and _train_proc.poll() is None:
            return jsonify(status="already_running", pid=_train_proc.pid)
        # drop any cached model so the fresh ckpt is picked up after training
        _state.pop("model", None)
        _state.pop("agent", None)
        _train_proc = subprocess.Popen(
            [sys.executable, "-u", os.path.join(ROOT, "train.py")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT,
        )
        return jsonify(status="started", pid=_train_proc.pid)


@app.post("/api/train/stop")
def api_train_stop():
    global _train_proc
    with _lock:
        if _train_proc is None or _train_proc.poll() is not None:
            return jsonify(status="not_running")
        _train_proc.send_signal(signal.SIGTERM)
        try:
            _train_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _train_proc.kill()
        return jsonify(status="stopped")


@app.get("/api/train/status")
def api_train_status():
    running = _train_proc is not None and _train_proc.poll() is None
    return jsonify(running=running, pid=_train_proc.pid if running else None)


@app.get("/api/architecture")
def api_architecture():
    model, cfg, device, _ = get_model()
    if model is None:
        # fall back to a freshly-instantiated model using the default cfg so the
        # architecture view works even before the first checkpoint is trained
        from model import LLM, ModelConfig  # local import to avoid cycles
        cfg = ModelConfig()
        model = LLM(cfg)
        device = "cpu"

    modules = []
    categories: dict[str, int] = {}

    def categorize(name: str) -> str:
        if "tok_emb" in name or "lm_head" in name:
            return "embedding"
        if ".attn." in name or name.endswith(".attn"):
            return "attention"
        if ".ffn." in name or name.endswith(".ffn"):
            return "ffn"
        if "norm" in name:
            return "norm"
        return "other"

    for name, p in model.named_parameters():
        cat = categorize(name)
        n = p.numel()
        categories[cat] = categories.get(cat, 0) + n
        modules.append({
            "name": name,
            "shape": list(p.shape),
            "params": n,
            "dtype": str(p.dtype).replace("torch.", ""),
            "category": cat,
        })

    # de-dup tied weight double-counting
    if cfg.tie_embeddings:
        for m in modules:
            if m["name"] == "lm_head.weight":
                categories["embedding"] -= m["params"]
                m["tied"] = True
                m["params"] = 0

    total = sum(categories.values())
    return jsonify({
        "config": cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg),
        "total_params": total,
        "total_params_m": round(total / 1e6, 2),
        "by_category": categories,
        "modules": modules,
        "repr": repr(model),
        "device": device,
    })


@app.get("/api/experiments")
def api_experiments():
    if not os.path.exists(EXPERIMENTS_PATH):
        return jsonify(ready=False)
    with open(EXPERIMENTS_PATH) as f:
        return jsonify(ready=True, **json.load(f))


@app.post("/api/generate")
def api_generate():
    payload = request.get_json(force=True)
    prompt = payload.get("prompt", "")
    max_tokens = int(payload.get("max_tokens", 80))
    temperature = float(payload.get("temperature", 0.8))
    model, cfg, device, enc = get_model()
    if model is None:
        return jsonify(error="no checkpoint — run train.py first"), 400
    ids = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens,
                             temperature=temperature, top_k=40)
    text = enc.decode(out[0].tolist())
    return jsonify(prompt=prompt, completion=text[len(prompt):], full=text)


@app.post("/api/agent")
def api_agent():
    payload = request.get_json(force=True)
    goal = payload.get("goal", "")
    result = get_agent().run(goal)
    return jsonify(result)


# ===================== Model Builder: custom experiments =====================

_exp_progress: dict = {"state": "idle"}
_exp_thread: threading.Thread | None = None


def _run_custom_thread(name: str, cfg_kwargs: dict, lr: float, seeds: list[int], steps: int):
    global _exp_progress
    try:
        from experiments import run_custom, append_result  # lazy import; heavy
        cfg = ModelConfig(**cfg_kwargs)
        _exp_progress = {
            "state": "running", "name": name, "n_seeds": len(seeds),
            "seeds": seeds, "steps": steps, "params_m": None,
            "seed_idx": 0, "step": 0, "loss": None,
        }

        def cb(stage, **kw):
            global _exp_progress
            if stage == "seed_start":
                _exp_progress.update(seed_idx=kw["seed_idx"], current_seed=kw["seed"], step=0, loss=None)
            elif stage == "step":
                _exp_progress.update(step=kw["step"], total=kw["total"], loss=round(kw["loss"], 4), lr=kw["lr"])
            elif stage == "seed_done":
                _exp_progress.update(last_seed_val_loss=round(kw["val_loss"], 4))
            elif stage == "all_done":
                _exp_progress["state"] = "saving"

        agg = run_custom(name, cfg, lr=lr, seeds=seeds, steps=steps, progress_cb=cb)
        append_result(agg, EXPERIMENTS_PATH)
        # drop cached model so the next /api/generate reloads fresh
        _state.pop("model", None)
        _exp_progress = {
            "state": "done", "name": name,
            "val_loss_mean": round(agg["val_loss"]["mean"], 4),
            "val_loss_std": round(agg["val_loss"]["std"], 4),
            "val_ppl_mean": round(agg["val_ppl"]["mean"], 2),
            "val_ppl_std": round(agg["val_ppl"]["std"], 2),
            "params_m": agg["params_m"],
            "seeds": seeds,
            "steps": steps,
        }
    except Exception as e:
        import traceback
        _exp_progress = {"state": "error", "error": str(e), "trace": traceback.format_exc()[-800:]}


@app.post("/api/experiment/run")
def api_experiment_run():
    global _exp_thread
    with _lock:
        if _exp_thread is not None and _exp_thread.is_alive():
            return jsonify(error="experiment already running"), 409
        payload = request.get_json(force=True) or {}
        cfg_kwargs = payload.get("config") or {}
        lr = float(payload.get("lr", 3e-4))
        seeds = payload.get("seeds") or [1, 2, 3]
        steps = int(payload.get("steps", 200))
        name = payload.get("name") or f"custom-{len(os.listdir(ROOT))}"
        _exp_thread = threading.Thread(
            target=_run_custom_thread, args=(name, cfg_kwargs, lr, seeds, steps), daemon=True,
        )
        _exp_thread.start()
        return jsonify(status="started", name=name, seeds=seeds, steps=steps)


@app.get("/api/experiment/progress")
def api_experiment_progress():
    return jsonify(_exp_progress)


# ===================== Details-on-demand (learning content) =====================

DETAILS = {
    "rms": {
        "title": "RMSNorm",
        "summary": "Root-Mean-Square normalization. Scales activations by the RMS along the feature dim, no mean-centering.",
        "used_in": ["LLaMA 1/2/3", "Gemma", "Qwen", "Mistral"],
        "tradeoffs": "Cheaper than LayerNorm (no mean subtraction). Empirically matches or beats LayerNorm at LLM scale.",
        "formula": "y = x / sqrt(mean(x^2) + eps) * gamma",
    },
    "layer": {
        "title": "LayerNorm",
        "summary": "Classic LayerNorm — subtract mean, divide by std, then affine (gamma, beta).",
        "used_in": ["GPT-2", "BERT", "original Transformer"],
        "tradeoffs": "Extra compute vs RMSNorm for no consistent quality gain at LLM scale.",
        "formula": "y = (x - mean(x)) / std(x) * gamma + beta",
    },
    "mha": {
        "title": "Multi-Head Attention (MHA)",
        "summary": "Each query head has its own key & value head. Most expressive, most memory.",
        "used_in": ["original Transformer", "GPT-2", "GPT-3"],
        "tradeoffs": "Highest quality but KV cache scales with n_heads → expensive for long-context inference.",
    },
    "gqa": {
        "title": "Grouped-Query Attention (GQA)",
        "summary": "Groups of query heads share a smaller number of KV heads (e.g. 8 Q heads → 2 KV heads).",
        "used_in": ["LLaMA 2 70B", "LLaMA 3", "Gemma 2", "Qwen 2"],
        "tradeoffs": "~4x smaller KV cache with <1% quality drop. Dominant choice for modern LLMs.",
    },
    "mqa": {
        "title": "Multi-Query Attention (MQA)",
        "summary": "All query heads share a single KV head. Max KV-cache compression.",
        "used_in": ["PaLM", "Falcon", "StarCoder"],
        "tradeoffs": "Smallest KV cache. Small quality regression; GQA is the usual compromise.",
    },
    "swiglu": {
        "title": "SwiGLU FFN",
        "summary": "Gated MLP: x → Swish(W1 x) ⊙ (W3 x) → W2. Two up-projections gated together.",
        "used_in": ["LLaMA", "PaLM", "Mistral"],
        "tradeoffs": "~0.5% lower loss than GELU at same params, 1.5x the up-projection weights. Worth it.",
    },
    "gelu": {
        "title": "GELU MLP",
        "summary": "Classic 2-layer MLP with GELU activation: W2(gelu(W1 x)). Hidden dim ≈ 4×d_model.",
        "used_in": ["GPT-2/3", "BERT"],
        "tradeoffs": "Simpler, fewer params than SwiGLU. Marginally worse at LLM scale.",
    },
    "pre": {
        "title": "Pre-norm residual",
        "summary": "Norm is applied inside the residual branch: x + Attn(Norm(x)). Stable, easy to train deep.",
        "used_in": ["GPT-2+", "LLaMA", "all modern LLMs"],
        "tradeoffs": "Strictly better than post-norm for >6 layers. Standard today.",
    },
    "post": {
        "title": "Post-norm residual",
        "summary": "Norm after the residual add: Norm(x + Attn(x)). Original Transformer formulation.",
        "used_in": ["original Transformer", "BERT"],
        "tradeoffs": "Hard to train deep (>12 layers) without warmup tricks. Generally avoided today.",
    },
    "rope": {
        "title": "Rotary Position Embeddings (RoPE)",
        "summary": "Rotates query and key vectors by angle θ·position. Injects position info multiplicatively.",
        "used_in": ["LLaMA", "GPT-NeoX", "PaLM", "Qwen"],
        "tradeoffs": "No learned params, extrapolates to longer contexts better than learned/sinusoidal.",
    },
    "learned": {
        "title": "Learned positional embeddings",
        "summary": "A learned vector for each position index, added to token embeddings.",
        "used_in": ["GPT-2", "BERT"],
        "tradeoffs": "Doesn't extrapolate past max_seq_len seen in training. Extra vocab*d_model params.",
    },
    "tied": {
        "title": "Tied embeddings",
        "summary": "Share the tok_embedding weight with the lm_head output projection.",
        "used_in": ["GPT-2", "most small LLMs"],
        "tradeoffs": "Saves vocab*d_model parameters (often 30–50% of total). Tiny quality cost at small scale.",
    },
    "seeds": {
        "title": "Multi-seed eval",
        "summary": "Train the same config with N different random seeds, report mean ± std. Filters out init luck.",
        "used_in": ["any rigorous ablation study"],
        "tradeoffs": "N× compute. Variance bands tell you which differences are real vs. noise — non-negotiable for small-scale comparisons.",
    },
    "dpo": {
        "title": "DPO — Direct Preference Optimization",
        "summary": "Fine-tune an LLM on preference pairs (chosen vs. rejected) without training a separate reward model.",
        "used_in": ["LLaMA 3 Instruct", "Mistral Instruct", "Zephyr"],
        "tradeoffs": "Simpler than RLHF (no PPO loop, no reward model). Needs a frozen reference policy. Cheaper and more stable.",
        "formula": "loss = -log σ(β · [log π(y+|x)/π_ref(y+|x) - log π(y-|x)/π_ref(y-|x)])",
    },
    "rlhf": {
        "title": "RLHF — Reinforcement Learning from Human Feedback",
        "summary": "3 stages: (1) SFT, (2) train reward model on preferences, (3) PPO fine-tune against the reward.",
        "used_in": ["InstructGPT", "ChatGPT original", "Claude"],
        "tradeoffs": "Flexible but complex: reward model overfitting, PPO instability, KL penalty tuning. DPO is now the default for most teams.",
    },
    "sft": {
        "title": "SFT — Supervised Fine-Tuning",
        "summary": "Standard next-token cross-entropy on high-quality instruction/response pairs.",
        "used_in": ["every instruct model, as the first alignment stage"],
        "tradeoffs": "Cheap, stable, but can't teach 'what not to say' — that's what DPO/RLHF adds on top.",
    },
    "bf16": {
        "title": "bfloat16 autocast",
        "summary": "16-bit float with 8-bit exponent (same range as fp32) and 7-bit mantissa. Used for matmul inside autocast region.",
        "used_in": ["all modern LLM training (A100+, H100, TPU, Apple Silicon MPS)"],
        "tradeoffs": "~2× faster than fp32, no manual loss scaling needed. Slightly worse than fp16 precision but no overflow issues.",
    },
}


@app.get("/api/details/<key>")
def api_details(key: str):
    d = DETAILS.get(key)
    if d is None:
        return jsonify(error="unknown key", key=key), 404
    return jsonify(d)


@app.get("/api/details")
def api_details_all():
    return jsonify(DETAILS)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Dashboard → http://localhost:{port}")
    print("To share publicly:  cloudflared tunnel --url http://localhost:" + str(port))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
