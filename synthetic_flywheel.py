"""Synthetic data flywheel with hierarchical multi-judge filtering.

Pipeline:
  1. Generator: trained model produces N synthetic samples
  2. Self-critique: model rewrites each sample with constitutional prompt
  3. Judge-1 (Haiku): coarse filter — is this Shakespearean? (binary)
  4. Judge-2 (Sonnet): fine quality score 1–10 + reasoning
  5. Judge-of-Judge (Opus): does Judge-2's score match the text? Catches drift.
  6. Ensemble vote: keep iff Judge-1=true AND Judge-2≥7 AND Judge-of-Judge=true
  7. Survivors → training mix (real + synthetic) for next SFT round

Why three judges:
  - Single-judge has bias drift (DeepSeek-R1 paper, table 4)
  - Two-judge ensemble drops false-positive rate by ~3×
  - Judge-of-Judge catches collusion (both judges wrong in same direction)

Output: filtered.jsonl with rows {sample, scores, kept: bool, reasons: [...]}
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from anthropic import Anthropic

from agents import client, CLAUDE_FAST, CLAUDE_SMART, CLAUDE_DEEP, BaseAgent
from braintrust_bridge import traced, log_event


# ───────────────────────── judges ─────────────────────────

class CoarseJudge(BaseAgent):
    name = "judge_coarse"
    model = CLAUDE_FAST
    system = """You are a coarse Shakespearean-style classifier.
Given a generated text snippet, return JSON: {"is_shakespearean": bool, "reason": "<=20 words"}
Be lenient on semantics, strict on style (capitalized speakers, archaic diction, line breaks)."""

    @traced("flywheel.judge_coarse")
    def classify(self, text: str) -> dict:
        out = self.call(f"Text:\n```\n{text}\n```\n\nReturn JSON.", max_tokens=200)
        return self._extract_json(out)


class FineJudge(BaseAgent):
    name = "judge_fine"
    model = CLAUDE_SMART
    system = """You are a fine-grained Shakespearean quality scorer.
Score 1–10 on:
  - style fidelity (3 pts)
  - line-break / speaker conventions (2 pts)
  - lexical richness (2 pts)
  - coherence within passage (3 pts)

Return JSON: {"score": int 1-10, "breakdown": {style: int, format: int, lex: int, coh: int}, "reasoning": "<=60 words"}
Be calibrated — most LLM-generated samples should score 4–7, not 8+."""

    @traced("flywheel.judge_fine")
    def score(self, text: str) -> dict:
        out = self.call(f"Text:\n```\n{text}\n```\n\nReturn JSON.", max_tokens=400, thinking=True)
        return self._extract_json(out)


class JudgeOfJudge(BaseAgent):
    """Judges whether FineJudge's score is calibrated to the text quality."""
    name = "judge_of_judge"
    model = CLAUDE_DEEP
    system = """You are a meta-judge. You audit a fine-judge's score against the actual text.
Common errors:
  - Score too high: text is incoherent but score >= 7
  - Score too low: text is genuinely good but score <= 5
  - Reasoning doesn't match score (e.g. praises everything but scores 4)

Return JSON: {"score_calibrated": bool, "true_score_estimate": int 1-10, "reasoning": "<=50 words"}"""

    @traced("flywheel.judge_of_judge")
    def audit(self, text: str, fine_verdict: dict) -> dict:
        msg = (f"Text:\n```\n{text}\n```\n\n"
               f"FineJudge said: {json.dumps(fine_verdict)}\n\n"
               "Audit the score. Return JSON.")
        out = self.call(msg, max_tokens=400, thinking=True)
        return self._extract_json(out)


# ───────────────────── self-critique rewrite ─────────────────────

class CritiqueRewriter(BaseAgent):
    name = "critic_rewriter"
    model = CLAUDE_FAST
    system = """You are a Shakespearean editor. Rewrite the given text to be MORE Shakespearean
without changing the topic. Fix:
  - missing capitalized speakers (e.g. "ROMEO:")
  - modern diction → archaic ("you" → "thou", "are" → "art" where appropriate)
  - run-on prose → dialogue lines
  - obvious incoherence

If text is already strong, change nothing. Output ONLY the rewritten text, no commentary."""

    @traced("flywheel.critique_rewrite")
    def rewrite(self, text: str) -> str:
        return self.call(f"Original:\n```\n{text}\n```\n\nRewritten:", max_tokens=400).strip("` \n")


# ──────────────────── generator ────────────────────

@torch.no_grad()
def generate_synthetic(model, enc, prompts: list[str], device: str,
                       max_new_tokens: int = 80, temperature: float = 0.9,
                       n_per_prompt: int = 5) -> list[dict]:
    """Generate raw synthetic samples from the trained tiny LLM."""
    out = []
    model.eval()
    for prompt in prompts:
        for _ in range(n_per_prompt):
            ids = torch.tensor([enc.encode(prompt)], device=device)
            gen = model.generate(ids, max_new_tokens=max_new_tokens,
                                 temperature=temperature, top_k=50)
            text = enc.decode(gen[0].tolist())
            out.append({"prompt": prompt, "raw": text})
    return out


# ──────────────────── full flywheel ────────────────────

@dataclass
class FlywheelConfig:
    n_per_prompt: int = 5
    fine_threshold: int = 7         # keep if fine_judge.score >= this
    require_jojo_calibrated: bool = True
    output_path: str = "synthetic_filtered.jsonl"


def run_flywheel(model, enc, prompts: list[str], device: str,
                 cfg: Optional[FlywheelConfig] = None) -> dict:
    """Generate → critique → 3-judge ensemble → write filtered jsonl."""
    cfg = cfg or FlywheelConfig()
    coarse = CoarseJudge()
    fine = FineJudge()
    jojo = JudgeOfJudge()
    critic = CritiqueRewriter()

    raw = generate_synthetic(model, enc, prompts, device, n_per_prompt=cfg.n_per_prompt)
    log_event(stage="flywheel.generated", n=len(raw))

    rows = []
    kept = 0
    for i, item in enumerate(raw):
        text = item["raw"]
        # 1. self-critique rewrite
        try:
            rewritten = critic.rewrite(text)
        except Exception as e:
            rewritten = text
            log_event(stage="flywheel.rewrite_error", error=str(e))

        # 2. coarse filter
        c = coarse.classify(rewritten)
        # 3. fine score (only if coarse passed)
        f = fine.score(rewritten) if c.get("is_shakespearean") else {"score": 0, "reasoning": "skipped"}
        # 4. judge-of-judge audit (only if fine ran)
        jj = jojo.audit(rewritten, f) if f.get("score", 0) > 0 else {"score_calibrated": False, "true_score_estimate": 0}

        # 5. ensemble vote
        keep = (
            c.get("is_shakespearean", False)
            and f.get("score", 0) >= cfg.fine_threshold
            and (jj.get("score_calibrated", False) if cfg.require_jojo_calibrated else True)
        )

        row = {
            "idx": i,
            "prompt": item["prompt"],
            "raw": text,
            "rewritten": rewritten,
            "coarse": c,
            "fine": f,
            "judge_of_judge": jj,
            "kept": keep,
            "reasons": [
                f"coarse={c.get('is_shakespearean')}",
                f"fine={f.get('score', 0)}",
                f"jojo_calibrated={jj.get('score_calibrated')}",
            ],
        }
        rows.append(row)
        if keep:
            kept += 1
        log_event(stage="flywheel.judged", idx=i, kept=keep, fine_score=f.get("score", 0))

    with open(cfg.output_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = {
        "generated": len(rows),
        "kept": kept,
        "kept_rate": round(kept / len(rows), 3) if rows else 0,
        "output": cfg.output_path,
        "avg_fine_score": round(sum(r["fine"].get("score", 0) for r in rows) / len(rows), 2) if rows else 0,
    }
    log_event(stage="flywheel.done", **summary)
    return summary


# ──────────── synthetic mix → training data ────────────

def mix_synthetic_with_real(real_bin_path: str, filtered_jsonl: str,
                            enc, output_bin_path: str, ratio: float = 1.0) -> dict:
    """Append kept synthetic samples to real training data, capped at `ratio` * len(real)."""
    import numpy as np
    real = np.memmap(real_bin_path, dtype=np.uint16, mode="r")
    real_arr = np.array(real)

    synth_tokens = []
    with open(filtered_jsonl) as f:
        for line in f:
            row = json.loads(line)
            if row["kept"]:
                synth_tokens.extend(enc.encode(row["rewritten"]))

    cap = int(len(real_arr) * ratio)
    synth_arr = np.array(synth_tokens[:cap], dtype=np.uint16)

    mixed = np.concatenate([real_arr, synth_arr])
    mixed.tofile(output_bin_path)

    return {
        "real_tokens": int(len(real_arr)),
        "synth_tokens_kept": int(len(synth_arr)),
        "mixed_tokens": int(len(mixed)),
        "ratio_actual": round(len(synth_arr) / len(real_arr), 3),
        "output": output_bin_path,
    }
