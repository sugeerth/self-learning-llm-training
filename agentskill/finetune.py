"""LoRA fine-tuning from curated trajectories — the GPU / Colab step.

'Use GPU sparingly': a small base model, low LoRA rank, gradient checkpointing,
a capped step count, and bf16. Most of the project's value (retrieval-augmented
imitation) needs no GPU at all — this distills the same curated experiences
into the weights when a GPU is available.

`build_sft_dataset()` is pure-Python and always runs (it's what the CLI's
`curate` command writes); `lora_finetune()` imports transformers/peft lazily
and raises a clear message if they (or a GPU) are absent, so importing this
module never requires torch.
"""

from __future__ import annotations

import json

from .scoring import curate
from .trajectory import Trajectory, canonical_for

SYSTEM = ("You are a capable agent. Given a task, output the tool sequence that "
          "solves it efficiently, one tool per line.")


def to_sft_example(t: Trajectory) -> dict:
    """One instruction/response pair: task -> the optimal tool plan. We teach
    the *canonical* plan for the trajectory's family (learn the skill, not the
    exploratory detours a successful-but-messy run happened to take)."""
    plan = canonical_for(t.domain, t.goal)
    prompt = (f"{SYSTEM}\n\nDomain: {t.domain}\nTask: {t.goal}\n"
              f"Available tools: known for this domain.\nPlan:")
    completion = "\n".join(plan)
    return {"prompt": prompt, "completion": completion,
            "domain": t.domain, "task_id": t.task_id}


def build_sft_dataset(trajs: list[Trajectory], min_quality: float = 0.6,
                      out_path: str | None = None) -> list[dict]:
    """Curate high-quality trajectories -> SFT pairs. Dedupe identical prompts."""
    seen, examples = set(), []
    for t, _score in curate(trajs, min_quality=min_quality):
        ex = to_sft_example(t)
        if ex["prompt"] in seen:
            continue
        seen.add(ex["prompt"])
        examples.append(ex)
    if out_path:
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
    return examples


def lora_finetune(sft_path: str, base_model: str = "sshleifer/tiny-gpt2",
                  out_dir: str = "runs/agentskill-lora", max_steps: int = 60,
                  lora_r: int = 8, batch_size: int = 4, lr: float = 2e-4):
    """Minimal LoRA SFT. Requires transformers + peft + a torch device.
    Kept deliberately small (tiny base, r=8, <=60 steps, grad checkpointing)
    to honour 'use GPU sparingly'. Returns the output dir on success."""
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   Trainer, TrainingArguments)
    except Exception as e:  # pragma: no cover - GPU/Colab only
        raise RuntimeError(
            "lora_finetune needs transformers+peft+torch (run on Colab GPU): "
            f"{e}") from e

    rows = [json.loads(l) for l in open(sft_path) if l.strip()]
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.pad_token = tok.pad_token or tok.eos_token

    def fmt(r):
        text = r["prompt"] + "\n" + r["completion"] + tok.eos_token
        enc = tok(text, truncation=True, max_length=256, padding="max_length")
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = Dataset.from_list(rows).map(fmt, remove_columns=["prompt", "completion",
                                                          "domain", "task_id"])
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, task_type="CAUSAL_LM"))
    args = TrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=batch_size,
        max_steps=max_steps, learning_rate=lr, logging_steps=10,
        bf16=torch.cuda.is_available(), report_to=[], save_strategy="no")
    Trainer(model=model, args=args, train_dataset=ds).train()
    model.save_pretrained(out_dir)
    return out_dir
