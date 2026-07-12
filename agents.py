"""Multi-agent hierarchy for self-learning LLM training.

Hierarchy:

    OrchestratorAgent  (top — picks next move)
        ├── TrainerAgent       — proposes next BlockSpec to try
        ├── EvaluatorAgent     — scores trained model on cloze + perplexity
        │     └── JudgeAgent           — audits the Evaluator's verdict
        │           └── MetaJudgeAgent — audits the Judge's reasoning
        └── HumanJudgeAgent    — emits a decision payload for human review

All calls go through Claude API with prompt caching, all wrapped in Braintrust traces.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

from anthropic import Anthropic

from braintrust_bridge import traced, log_event

CLAUDE_FAST = "claude-haiku-4-5-20251001"   # cheap, fast — trainer/evaluator
CLAUDE_SMART = "claude-sonnet-4-6"          # judges
CLAUDE_DEEP = "claude-opus-4-7"             # meta-judge + orchestrator (rare calls)

# Phase 3 (model-onramp): agents resolve their model by ROLE through the
# capability router when probed manifests exist; the constants above remain
# the fallback so nothing changes until models are onboarded.
try:
    from onramp_bridge import record_outcome, record_quality, resolve_model
except Exception:  # bridge or model-onramp absent — keep legacy behavior
    def resolve_model(role: str, default: str) -> str:
        return default

    def record_outcome(*args, **kwargs) -> None:
        pass

    def record_quality(*args, **kwargs) -> None:
        pass

_client: Optional[Anthropic] = None


def client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


# ────────────────────────────── shared types ──────────────────────────────

@dataclass
class TrainProposal:
    name: str
    config: dict
    lr: float
    rationale: str  # why this config is worth trying


@dataclass
class EvalReport:
    val_loss: float
    val_ppl: float
    cloze_accuracy: float
    sample_quality: int    # 1–10 self-rated
    notes: str


@dataclass
class JudgeVerdict:
    accept: bool
    confidence: float       # 0..1
    flagged: list[str]      # specific issues
    reasoning: str


@dataclass
class MetaVerdict:
    judge_was_correct: bool
    bias_detected: Optional[str]   # e.g. "lenient on overfitting"
    reasoning: str


# ────────────────────────────── base agent ──────────────────────────────

class BaseAgent:
    """All agents share: cached system prompt + traced calls + structured output."""
    name: str = "base"
    model: str = CLAUDE_FAST   # fallback when the on-ramp has no probed model
    role: str = ""             # onramp role profile; "" = always use `model`
    system: str = ""
    last_model: str = ""       # model that served the most recent call

    def call(self, user_msg: str, max_tokens: int = 1024, thinking: bool = False) -> str:
        model = resolve_model(self.role, self.model) if self.role else self.model
        self.last_model = model
        msgs = [{"role": "user", "content": user_msg}]
        sys = [{
            "type": "text",
            "text": self.system,
            "cache_control": {"type": "ephemeral"},  # prompt caching — system is reused
        }]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": sys,
            "messages": msgs,
        }
        if thinking and model.startswith(("claude-opus", "claude-sonnet", "claude-fable")):
            # adaptive is the only thinking mode on Opus 4.7+/Sonnet 5;
            # budget_tokens is rejected there
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["max_tokens"] = max_tokens + 2048

        t0 = time.time()
        try:
            resp = client().messages.create(**kwargs)
        except Exception:
            # live failure feeds the on-ramp's circuit breaker + autopilot
            record_outcome(self.role, model, success=False,
                           latency_s=time.time() - t0)
            raise
        record_outcome(self.role, model, success=True,
                       input_tokens=resp.usage.input_tokens,
                       output_tokens=resp.usage.output_tokens,
                       latency_s=time.time() - t0)

        # extract text (skip thinking blocks)
        out = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        log_event(
            agent=self.name,
            model=model,
            input_tokens=resp.usage.input_tokens,
            cache_read=getattr(resp.usage, "cache_read_input_tokens", 0),
            cache_write=getattr(resp.usage, "cache_creation_input_tokens", 0),
            output_tokens=resp.usage.output_tokens,
            user_msg=user_msg[:500],
            response=out[:500],
        )
        return out

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Pull the last fenced JSON block, or first {...} span."""
        if "```json" in text:
            chunk = text.split("```json")[-1].split("```")[0]
        elif "```" in text:
            chunk = text.split("```")[1]
        else:
            start = text.find("{")
            end = text.rfind("}")
            chunk = text[start : end + 1] if start >= 0 and end > start else "{}"
        return json.loads(chunk.strip())


# ────────────────────────────── trainer ──────────────────────────────

class TrainerAgent(BaseAgent):
    name = "trainer"
    model = CLAUDE_FAST
    role = "trainer"
    system = """You are the TrainerAgent in a self-learning LLM training loop.
You propose the next architecture configuration to try given prior sweep results.

Search space (LLaMA-style decoder):
- d_model: [128, 192, 256, 320, 384, 448, 512, 640]
- n_layers: [2, 4, 6, 8, 10]
- n_heads: divisor of d_model, in [2, 4, 6, 8]
- n_kv_heads: 1 (MQA), 2 (GQA), or n_heads (MHA)
- d_ff_mult: 2.0 .. 4.0
- lr: log-uniform in [1e-4, 1e-3]
- tie_embeddings: usually true (untied wastes compute at small scale)

Always respond with EXACTLY one JSON block in ```json fences containing:
  {"name": "...", "config": {...}, "lr": float, "rationale": "<= 50 words"}
Avoid duplicating prior names. Lean toward configs the priors suggest are under-explored
near the current Pareto frontier (params vs val_ppl)."""

    @traced("trainer.propose")
    def propose(self, history: list[dict]) -> TrainProposal:
        msg = (
            "Prior sweep results (sorted best→worst by val_ppl):\n"
            + json.dumps(history, indent=2)
            + "\n\nPropose ONE new config that is likely to beat the current best."
        )
        text = self.call(msg, max_tokens=600)
        d = self._extract_json(text)
        return TrainProposal(**d)


# ────────────────────────────── evaluator ──────────────────────────────

class EvaluatorAgent(BaseAgent):
    name = "evaluator"
    model = CLAUDE_FAST
    role = "evaluator"
    system = """You are the EvaluatorAgent. You score a trained tiny LLM's outputs.
You receive: val_loss, val_ppl, cloze_accuracy, sample text.

Return JSON:
  {"val_loss": float, "val_ppl": float, "cloze_accuracy": float,
   "sample_quality": int (1-10), "notes": "<= 60 words"}

Rate sample_quality on coherence + style fidelity to Shakespeare, NOT semantics
(nonsense is expected at 200 steps). Be calibrated, not generous."""

    @traced("evaluator.score")
    def score(self, raw_metrics: dict, sample: str) -> EvalReport:
        msg = (
            f"Raw metrics:\n{json.dumps(raw_metrics, indent=2)}\n\n"
            f"Generated sample (prompt='ROMEO:\\n', temp=0.8):\n```\n{sample}\n```\n\n"
            "Return your eval JSON."
        )
        text = self.call(msg, max_tokens=400)
        return EvalReport(**self._extract_json(text))


# ────────────────────────────── judge ──────────────────────────────

class JudgeAgent(BaseAgent):
    name = "judge"
    model = CLAUDE_SMART
    role = "judge"
    system = """You are the JudgeAgent. You audit the EvaluatorAgent's verdict and
either ACCEPT or REJECT it. You are skeptical, not deferential.

Common evaluator failures to flag:
  - sample_quality inflated relative to val_ppl
  - val_ppl reported but cloze_accuracy ignored
  - notes contradict the numbers
  - obvious overfitting/divergence not noted
  - benchmark prompt cherry-picked

Return JSON:
  {"accept": bool, "confidence": float (0-1), "flagged": [str], "reasoning": "<=80 words"}"""

    @traced("judge.audit")
    def audit(self, proposal: TrainProposal, report: EvalReport, sample: str) -> JudgeVerdict:
        msg = (
            f"Proposal: {asdict(proposal)}\n"
            f"Evaluator report: {asdict(report)}\n"
            f"Sample: ```\n{sample}\n```\n\n"
            "Audit the report. Return JSON verdict."
        )
        text = self.call(msg, max_tokens=500, thinking=True)
        return JudgeVerdict(**self._extract_json(text))


# ────────────────────────────── meta-judge ──────────────────────────────

class MetaJudgeAgent(BaseAgent):
    name = "meta_judge"
    model = CLAUDE_DEEP
    role = "meta_judge"
    system = """You are the MetaJudgeAgent — the judge of judges. You audit the
JudgeAgent's reasoning over time, looking for systematic bias:

  - Lenient bias: accepts borderline runs that should be rejected
  - Strict bias: rejects runs that match historical accept patterns
  - Anchoring: judge over-weights the first metric in the report
  - Sample-blindness: ignores qualitative sample issues

Return JSON:
  {"judge_was_correct": bool, "bias_detected": str|null, "reasoning": "<=80 words"}"""

    @traced("meta_judge.audit")
    def audit(self, judge_history: list[dict], current: dict) -> MetaVerdict:
        msg = (
            f"Judge's last 10 verdicts:\n{json.dumps(judge_history[-10:], indent=2)}\n\n"
            f"Current verdict to audit:\n{json.dumps(current, indent=2)}\n\n"
            "Return your meta-verdict JSON."
        )
        text = self.call(msg, max_tokens=500, thinking=True)
        d = self._extract_json(text)
        if "bias_detected" not in d:
            d["bias_detected"] = None
        return MetaVerdict(**d)


# ────────────────────────────── human-judge surface ──────────────────────────────

class HumanJudgeAgent(BaseAgent):
    """Doesn't call an LLM — emits a payload for a human to decide on."""
    name = "human_judge"

    @staticmethod
    @traced("human_judge.request")
    def request_decision(reason: str, context: dict) -> dict:
        decision = {
            "ts": time.time(),
            "reason": reason,
            "context": context,
            "status": "pending",
        }
        # writes to /tmp/llm_training_human_queue.jsonl — dashboard polls this
        with open("/tmp/llm_training_human_queue.jsonl", "a") as f:
            f.write(json.dumps(decision) + "\n")
        return decision


# ────────────────────────────── orchestrator ──────────────────────────────

class OrchestratorAgent(BaseAgent):
    """Top-level loop. Owns the hierarchy and decides when to escalate to humans."""
    name = "orchestrator"
    role = "orchestrator"

    def __init__(self) -> None:
        self.trainer = TrainerAgent()
        self.evaluator = EvaluatorAgent()
        self.judge = JudgeAgent()
        self.meta = MetaJudgeAgent()
        self.judge_log: list[dict] = []

    @traced("orchestrator.step")
    def step(self, history: list[dict], train_fn) -> dict:
        """One full self-learning step.
        train_fn(config, lr) -> {raw_metrics, sample}
        """
        proposal = self.trainer.propose(history)
        log_event(stage="proposal", proposal=asdict(proposal))

        run = train_fn(proposal.config, proposal.lr)
        log_event(stage="train_done", metrics=run["raw_metrics"])

        report = self.evaluator.score(run["raw_metrics"], run["sample"])
        verdict = self.judge.audit(proposal, report, run["sample"])
        self.judge_log.append({
            "proposal": asdict(proposal),
            "report": asdict(report),
            "verdict": asdict(verdict),
        })

        # the judge's verdict on the evaluator's report doubles as a quality
        # signal for whichever model served the evaluator (onramp autopilot
        # reads these when deciding promotions)
        if self.evaluator.last_model:
            record_quality("evaluator", self.evaluator.last_model,
                           verdict.confidence if verdict.accept
                           else 1.0 - verdict.confidence)

        meta = self.meta.audit(self.judge_log, self.judge_log[-1])

        # likewise, the meta-judge's audit scores the judge's model
        if self.judge.last_model:
            record_quality("judge", self.judge.last_model,
                           1.0 if meta.judge_was_correct else 0.0)

        # escalate to human if meta-judge flags bias OR confidence is low
        if meta.bias_detected or verdict.confidence < 0.4:
            HumanJudgeAgent.request_decision(
                reason=meta.bias_detected or "low judge confidence",
                context={"proposal": asdict(proposal), "report": asdict(report),
                         "verdict": asdict(verdict), "meta": asdict(meta)},
            )

        return {
            "proposal": asdict(proposal),
            "report": asdict(report),
            "verdict": asdict(verdict),
            "meta": asdict(meta),
            "metrics": run["raw_metrics"],
            "sample": run["sample"],
            "ts": time.time(),
        }
