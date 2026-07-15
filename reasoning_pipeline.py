"""SOTA small-reasoning-model pipeline (early 2026).

Combines techniques proven on sub-2B reasoning models:
  - DeepSeek-R1-Distill          → GRPO + distillation from R1 traces
  - rStar-Math (Microsoft)        → MCTS rollouts at TRAINING time + Process Reward Model
  - Phi-4-mini-reasoning          → curriculum learning, reflection tokens
  - Qwen2.5-Math + Math-Shepherd  → step-level PRM with online DPO
  - SmolLM2-360M                  → trained 4T tokens, beats 1B models — small data quality > size

Key idea: a 1B-param model can match GPT-4o on math IF you train it with:
  1. PRM-supervised (per-step) reward instead of outcome-only reward (ORM)
  2. MCTS-expanded reasoning traces (search at train time, not just inference)
  3. Verifiable rewards (math, code) — checkable, no judge bias
  4. Reflection tokens (<think>...</think>) — explicit CoT scaffolding
  5. Self-correction loop (model fixes own mistakes)

This module provides each piece as a clean class. The base substrate is either
the existing model.py LLM (tiny) OR a HuggingFace AutoModelForCausalLM (1B class).
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from braintrust_bridge import traced, log_event


# ════════════════════════════ reflection token scaffold ════════════════════════════

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"


def reflection_format(question: str) -> str:
    """Format a prompt to elicit reflection-style reasoning."""
    return (
        f"Question: {question}\n"
        f"{THINK_OPEN}\n"
        # model fills in CoT here
    )


def parse_reflection(text: str) -> dict:
    """Extract think + answer from a reflection-formatted response."""
    think_m = re.search(rf"{THINK_OPEN}(.*?){THINK_CLOSE}", text, re.DOTALL)
    ans_m = re.search(rf"{ANSWER_OPEN}(.*?){ANSWER_CLOSE}", text, re.DOTALL)
    return {
        "think": think_m.group(1).strip() if think_m else "",
        "answer": ans_m.group(1).strip() if ans_m else "",
        "raw": text,
    }


# ════════════════════════════ verifiable rewards ════════════════════════════

class VerifiableReward:
    """Rewards that are CHECKABLE — no LLM judge needed.
    Eliminates judge bias entirely. Used by DeepSeek-R1, Qwen2.5-Math.
    """

    @staticmethod
    def math_answer(generated: str, gold: str) -> float:
        """Extract last number, compare to gold answer (GSM8K convention)."""
        ans = parse_reflection(generated)["answer"]
        if not ans:
            ans = generated.split()[-1] if generated.split() else ""
        # extract a number, possibly with $ or commas. Commas are stripped first,
        # so we match a contiguous digit run: the old comma-grouped pattern capped the
        # integer part at 3 digits and split numbers >= 1000 (e.g. "1234" -> "4").
        gen_num = re.findall(r"-?\$?\d+(?:\.\d+)?", ans.replace(",", ""))
        gold_num = re.findall(r"-?\d+(?:\.\d+)?", gold.replace(",", ""))
        if not gen_num or not gold_num:
            return 0.0
        try:
            return 1.0 if abs(float(gen_num[-1].replace("$", "")) - float(gold_num[-1])) < 1e-3 else 0.0
        except ValueError:
            return 0.0

    @staticmethod
    def code_passes_tests(generated_code: str, test_cases: list[tuple[str, str]]) -> float:
        """Run generated python through test cases. Reward = fraction passing."""
        passed = 0
        for inp, expected in test_cases:
            try:
                # SAFETY: in production sandbox this with subprocess + timeout + restricted env
                local = {}
                exec(generated_code, {"__builtins__": __builtins__}, local)
                if "solve" in local:
                    out = local["solve"](inp)
                    if str(out).strip() == str(expected).strip():
                        passed += 1
            except Exception:
                pass
        return passed / max(len(test_cases), 1)

    @staticmethod
    def format_compliance(text: str) -> float:
        """Reward proper use of <think>/<answer> tags. Used as auxiliary reward."""
        has_think = THINK_OPEN in text and THINK_CLOSE in text
        has_answer = ANSWER_OPEN in text and ANSWER_CLOSE in text
        ordered = text.find(THINK_OPEN) < text.find(ANSWER_OPEN) if (has_think and has_answer) else False
        return float(has_think) * 0.3 + float(has_answer) * 0.4 + float(ordered) * 0.3


# ════════════════════════════ Process Reward Model ════════════════════════════

@dataclass
class ReasoningStep:
    text: str
    step_idx: int
    score: Optional[float] = None    # PRM score
    is_terminal: bool = False


class ProcessRewardModel:
    """PRM: scores each reasoning STEP, not just the final answer.

    Two flavors supported:
      - Trained PRM: separate small model fine-tuned on (step, correctness) pairs
      - Judge-LLM PRM: zero-shot Claude/GPT scores each step

    Math-Shepherd (Wang 2024) showed PRM > ORM (Outcome RM) for math by ~6 abs pts.
    rStar-Math (Microsoft 2025) made it work with MCTS at TRAINING time.
    """

    def __init__(self, mode: str = "judge", judge_agent=None):
        assert mode in ("judge", "trained")
        self.mode = mode
        self.judge = judge_agent

    @staticmethod
    def split_into_steps(text: str) -> list[ReasoningStep]:
        """Split CoT by 'Step N:' or sentence boundaries or newlines."""
        # try numbered steps first
        m = re.split(r"\n\s*Step\s+\d+[:\.]\s*", text)
        if len(m) > 1:
            return [ReasoningStep(text=s.strip(), step_idx=i) for i, s in enumerate(m) if s.strip()]
        # fall back to double-newlines
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [ReasoningStep(text=p, step_idx=i) for i, p in enumerate(parts)]

    @traced("prm.score_steps")
    def score_steps(self, question: str, steps: list[ReasoningStep], gold_answer: Optional[str] = None) -> list[ReasoningStep]:
        """Assign per-step score. Returns same list with .score filled."""
        if self.mode == "judge":
            return self._judge_score(question, steps, gold_answer)
        return self._trained_score(question, steps)

    def _judge_score(self, question: str, steps: list[ReasoningStep], gold: Optional[str]) -> list[ReasoningStep]:
        """Use Claude to score each step on correctness + progress."""
        if self.judge is None:
            from agents import BaseAgent, CLAUDE_SMART
            class _PRMJudge(BaseAgent):
                name = "prm_judge"
                model = CLAUDE_SMART
                system = """You score reasoning steps for a math/code problem.
For each step you assess: is it correct? does it make progress toward the goal?
Return JSON: {"scores": [float between 0 and 1, one per step], "reasoning": "<=60 words"}"""
            self.judge = _PRMJudge()

        msg = (
            f"Question: {question}\n"
            f"Gold answer: {gold or '(unknown)'}\n\n"
            "Steps:\n" + "\n".join(f"[{s.step_idx}] {s.text}" for s in steps) +
            "\n\nReturn JSON with one score per step."
        )
        out = self.judge.call(msg, max_tokens=600, thinking=True)
        d = self.judge._extract_json(out)
        scores = d.get("scores", [0.5] * len(steps))
        for s, sc in zip(steps, scores):
            s.score = float(sc)
        return steps

    def _trained_score(self, question: str, steps: list[ReasoningStep]) -> list[ReasoningStep]:
        """Stub — when a trained PRM (small classifier head) is available, plug it here."""
        for s in steps:
            s.score = 0.5  # uninformative prior
        return steps


# ════════════════════════════ MCTS at training time ════════════════════════════

@dataclass
class MCTSNode:
    """Node in the reasoning-step tree."""
    step_text: str
    parent: Optional["MCTSNode"] = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 1.0       # PRM-predicted quality of this step

    @property
    def avg_value(self) -> float:
        return self.value_sum / max(self.visits, 1)

    def ucb(self, c: float = 1.4) -> float:
        if not self.parent:
            return self.avg_value
        return self.avg_value + c * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)

    def trace(self) -> list["MCTSNode"]:
        path = []
        cur = self
        while cur:
            path.append(cur)
            cur = cur.parent
        return list(reversed(path))


class MCTSReasoner:
    """rStar-Math-style MCTS at training time.

    For each problem:
      1. Build a search tree where each node is a reasoning step
      2. Expansion: model proposes K next-step candidates
      3. Evaluation: PRM scores each leaf
      4. Backpropagation: update parent values
      5. After N rollouts, harvest top-K paths as training data
      6. Each node is also a (state, action, reward) tuple for the policy

    This converts inference-time search into training-time data.
    """

    def __init__(self, generator: Callable, prm: ProcessRewardModel,
                 max_depth: int = 8, branching: int = 3, rollouts: int = 20):
        self.gen = generator           # generator(prefix, n_samples) -> list[str]
        self.prm = prm
        self.max_depth = max_depth
        self.branching = branching
        self.rollouts = rollouts

    @traced("mcts.search")
    def search(self, question: str, gold_answer: Optional[str] = None) -> dict:
        root = MCTSNode(step_text=f"Question: {question}", prior=1.0)

        for r in range(self.rollouts):
            # 1. SELECT — descend by UCB
            node = root
            while node.children:
                node = max(node.children, key=lambda c: c.ucb())
                if len(node.trace()) - 1 >= self.max_depth:
                    break

            # 2. EXPAND — propose B next steps
            if node.visits > 0 and len(node.trace()) - 1 < self.max_depth:
                prefix = "\n".join(n.step_text for n in node.trace()[1:]) or ""
                candidates = self.gen(question + "\n" + prefix, self.branching)
                steps = [ReasoningStep(text=c, step_idx=len(node.trace())) for c in candidates]
                steps = self.prm.score_steps(question, steps, gold_answer)
                for s in steps:
                    child = MCTSNode(step_text=s.text, parent=node, prior=s.score or 0.5)
                    node.children.append(child)
                if node.children:
                    node = node.children[0]

            # 3. EVALUATE — PRM scores this leaf in isolation
            leaf_step = ReasoningStep(text=node.step_text, step_idx=len(node.trace()) - 1)
            scored = self.prm.score_steps(question, [leaf_step], gold_answer)
            value = scored[0].score or 0.5

            # 4. BACKPROP
            cur = node
            while cur:
                cur.visits += 1
                cur.value_sum += value
                cur = cur.parent

        # harvest top paths as training data
        paths = self._harvest_paths(root, top_k=3)
        return {
            "root_visits": root.visits,
            "tree_size": self._count_nodes(root),
            "best_path_value": paths[0]["avg_value"] if paths else 0,
            "paths": paths,
        }

    def _harvest_paths(self, root: MCTSNode, top_k: int) -> list[dict]:
        """Find the top-k highest-value paths from root to leaf."""
        leaves = []
        def walk(n):
            if not n.children:
                leaves.append(n)
            for c in n.children:
                walk(c)
        walk(root)
        leaves.sort(key=lambda l: -l.avg_value)
        return [{
            "steps": [n.step_text for n in l.trace()[1:]],
            "avg_value": l.avg_value,
            "depth": len(l.trace()) - 1,
            "visits": l.visits,
        } for l in leaves[:top_k]]

    def _count_nodes(self, root: MCTSNode) -> int:
        n = 1
        for c in root.children:
            n += self._count_nodes(c)
        return n


# ════════════════════════════ self-correction loop ════════════════════════════

class SelfCorrector:
    """Model critiques its own answer, then rewrites if confidence is low.
    Used in OpenAI's o1-mini and Phi-4-mini-reasoning. Adds ~3-5pp on math benchmarks
    at the cost of 2× inference compute (the self-critique pass).
    """

    def __init__(self, generator: Callable, critic_judge=None):
        self.gen = generator
        self.critic = critic_judge

    @traced("self_correct.loop")
    def correct(self, question: str, max_iters: int = 2) -> dict:
        from agents import BaseAgent, CLAUDE_SMART
        if self.critic is None:
            class _Critic(BaseAgent):
                name = "self_critic"
                model = CLAUDE_SMART
                system = """You critique a model's reasoning.
Identify: (a) factual errors, (b) logical gaps, (c) calculation mistakes.
Return JSON: {"errors": [str], "needs_revision": bool, "confidence": float 0-1}"""
            self.critic = _Critic()

        history = []
        current = self.gen(reflection_format(question), 1)[0]
        history.append({"iter": 0, "text": current})

        for i in range(max_iters):
            crit_out = self.critic.call(
                f"Question: {question}\n\nCandidate answer:\n{current}\n\nReturn JSON critique.",
                max_tokens=400,
            )
            crit = self.critic._extract_json(crit_out)
            history.append({"iter": i + 1, "critique": crit})
            if not crit.get("needs_revision"):
                break
            errors_str = "\n".join(f"- {e}" for e in crit.get("errors", []))
            revise_prompt = (
                f"Question: {question}\n\n"
                f"Previous answer:\n{current}\n\n"
                f"Errors found:\n{errors_str}\n\n"
                f"Rewrite the reasoning AND answer, fixing these errors:\n"
                f"{THINK_OPEN}\n"
            )
            current = self.gen(revise_prompt, 1)[0]
            history.append({"iter": i + 1, "revised": current})

        return {"final": current, "iterations": len(history) // 2, "history": history}


# ════════════════════════════ teacher distillation ════════════════════════════

class R1Distiller:
    """Distill reasoning traces from a strong teacher (Claude opus / DeepSeek-R1)
    into a small student. The student learns to mimic the teacher's CoT, not just answer.

    DeepSeek-R1-Distill recipe:
      1. Teacher generates 800k math+code reasoning traces (verifiable correct ones)
      2. Student SFT on (question, full_trace)
      3. Optional: GRPO on top with verifiable rewards

    R1-Distill-Qwen-1.5B beats GPT-4o on AIME with this recipe.
    """

    def __init__(self, teacher_call: Callable, problems: list[dict]):
        self.teacher = teacher_call
        self.problems = problems  # [{"question", "answer", "test_cases"}]

    @traced("distill.collect_traces")
    def collect_traces(self, n: int = 100, only_correct: bool = True) -> list[dict]:
        """Generate teacher traces, filter to verifiably correct ones."""
        traces = []
        for problem in self.problems[:n]:
            prompt = (
                f"Solve this problem step by step. Use {THINK_OPEN}...{THINK_CLOSE} for reasoning "
                f"and {ANSWER_OPEN}...{ANSWER_CLOSE} for the final answer.\n\n"
                f"Question: {problem['question']}"
            )
            trace = self.teacher(prompt)
            if only_correct:
                if "answer" in problem:
                    r = VerifiableReward.math_answer(trace, problem["answer"])
                    if r < 1.0:
                        continue
                elif "test_cases" in problem:
                    r = VerifiableReward.code_passes_tests(parse_reflection(trace)["answer"], problem["test_cases"])
                    if r < 1.0:
                        continue
            traces.append({
                "question": problem["question"],
                "trace": trace,
                "answer": problem.get("answer"),
                "ts": time.time(),
            })
        return traces


# ════════════════════════════ end-to-end pipeline driver ════════════════════════════

@dataclass
class ReasoningPipelineConfig:
    base_model: str = "tiny"           # "tiny" | "smollm2-360m" | "llama-3.2-1b"
    use_mcts: bool = True
    use_prm: bool = True
    use_self_correction: bool = True
    use_distillation: bool = False     # requires teacher API budget
    rollouts_per_problem: int = 20
    branching_factor: int = 3
    max_reasoning_depth: int = 8


def end_to_end_reasoning_step(
    question: str,
    gold_answer: Optional[str],
    cfg: ReasoningPipelineConfig,
    generator: Callable,
) -> dict:
    """Run one full reasoning step end-to-end. This is the unit of training data.

    Returns the harvested MCTS paths + verifiable reward + self-correction trace,
    all of which become training signal for the next round.
    """
    prm = ProcessRewardModel(mode="judge") if cfg.use_prm else None
    out = {"question": question, "gold": gold_answer}

    if cfg.use_mcts and prm is not None:
        mcts = MCTSReasoner(generator, prm,
                            max_depth=cfg.max_reasoning_depth,
                            branching=cfg.branching_factor,
                            rollouts=cfg.rollouts_per_problem)
        out["mcts"] = mcts.search(question, gold_answer)

    if cfg.use_self_correction:
        sc = SelfCorrector(generator)
        out["self_corrected"] = sc.correct(question, max_iters=2)

    # verifiable reward on best path
    best_text = (out.get("self_corrected", {}).get("final")
                 or "\n".join(out.get("mcts", {}).get("paths", [{}])[0].get("steps", []))
                 or generator(question, 1)[0])
    if gold_answer:
        out["verifiable_reward"] = VerifiableReward.math_answer(best_text, gold_answer)
    out["format_reward"] = VerifiableReward.format_compliance(best_text)
    log_event(stage="reasoning_step", question=question[:100], reward=out.get("verifiable_reward", 0))
    return out
