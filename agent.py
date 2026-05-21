"""Agentic wrapper around the trained LLM.

The trained model is tiny and won't reliably emit structured JSON, so the
agent uses a HYBRID design:

  1. A deterministic parser/planner decomposes the user request into tool
     calls (this is the "brain" a 10M model can't be).
  2. The LLM provides the *creative* completion — writing in the voice it
     learned during training (Shakespeare).
  3. Tools give the agent real capabilities: calculator, word-count, search
     over its own training corpus, and LLM-backed text generation.

This mirrors production agent loops (tool definitions → LLM proposes call →
harness executes → result fed back) but with a rule-based proposer so we can
demo it with a toy model. Swap the proposer for a real LLM and the harness
stays the same.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

import torch

from data import prepare, tokenizer
from model import LLM, ModelConfig
from train import CKPT_PATH, pick_device

# ---------- tools ----------

_CORPUS: str | None = None


def _corpus() -> str:
    global _CORPUS
    if _CORPUS is None:
        txt = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare.txt")
        with open(txt, "r", encoding="utf-8") as f:
            _CORPUS = f.read()
    return _CORPUS


def tool_calculator(expr: str) -> str:
    # safe eval: only digits, operators, parens, decimal
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return "error: only arithmetic allowed"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {e}"


def tool_word_count(text: str) -> str:
    return str(len(text.split()))


def tool_search_corpus(query: str, k: int = 3) -> str:
    """Find lines in the training data containing the query (case-insensitive)."""
    lines = _corpus().splitlines()
    q = query.lower()
    hits = [ln for ln in lines if q in ln.lower()][:k]
    return "\n".join(hits) if hits else "no matches"


@dataclass
class LLMTool:
    model: LLM
    enc: Any
    device: str

    def __call__(self, prompt: str, max_tokens: int = 60, temperature: float = 0.8) -> str:
        ids = torch.tensor([self.enc.encode_ordinary(prompt)], dtype=torch.long, device=self.device)
        out = self.model.generate(ids, max_new_tokens=max_tokens,
                                  temperature=temperature, top_k=40)
        return self.enc.decode(out[0].tolist())[len(prompt):]


# ---------- agent ----------

@dataclass
class Step:
    thought: str
    tool: str
    args: dict
    result: str


class Agent:
    def __init__(self, load_llm: bool = True):
        self.device, _ = pick_device()
        self.enc = tokenizer()
        self.tools: dict[str, Callable[..., str]] = {
            "calculator": tool_calculator,
            "word_count": tool_word_count,
            "search_corpus": tool_search_corpus,
        }
        if load_llm and os.path.exists(CKPT_PATH):
            prepare()  # ensure corpus file exists
            ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)
            cfg = ModelConfig(**ckpt["cfg"])
            m = LLM(cfg).to(self.device)
            m.load_state_dict(ckpt["model"])
            m.eval()
            self.tools["llm_complete"] = LLMTool(m, self.enc, self.device)

    def plan(self, goal: str) -> list[tuple[str, dict]]:
        """Rule-based proposer → sequence of (tool, args)."""
        g = goal.lower()
        plan: list[tuple[str, dict]] = []
        # calculator
        m = re.search(r"([-+*/().\d\s]{3,})", goal)
        if m and any(op in m.group(1) for op in "+-*/") and any(c.isdigit() for c in m.group(1)):
            plan.append(("calculator", {"expr": m.group(1).strip()}))
        # search
        m = re.search(r"(?:find|search|look up|quote[s]? (?:about|on|with))\s+['\"]?([^'\"?.!]+)", g)
        if m:
            plan.append(("search_corpus", {"query": m.group(1).strip()}))
        # word count
        m = re.search(r"word count (?:of|for)?\s*['\"](.+?)['\"]", goal)
        if m:
            plan.append(("word_count", {"text": m.group(1)}))
        # creative completion
        if any(w in g for w in ("write", "continue", "complete", "verse", "poem", "soliloquy")):
            seed = re.sub(r"^(write|continue|complete)\s+(a\s+)?", "", goal, flags=re.I).strip(" .?!\"'")
            prompt = seed or "ROMEO:\n"
            plan.append(("llm_complete", {"prompt": prompt, "max_tokens": 80}))
        if not plan:
            # default: let the LLM riff on it
            if "llm_complete" in self.tools:
                plan.append(("llm_complete", {"prompt": goal, "max_tokens": 60}))
        return plan

    def run(self, goal: str) -> dict:
        trace: list[Step] = []
        steps = self.plan(goal)
        for tool, args in steps:
            fn = self.tools.get(tool)
            if fn is None:
                trace.append(Step(f"want to call {tool} but unavailable", tool, args, "SKIPPED"))
                continue
            result = fn(**args)
            trace.append(Step(f"calling {tool}({args})", tool, args, result))
        # synthesize final answer
        final = "\n\n".join(f"[{s.tool}] → {s.result}" for s in trace) or "(no steps)"
        return {
            "goal": goal,
            "trace": [s.__dict__ for s in trace],
            "answer": final,
        }


def main():
    agent = Agent()
    print("tools:", list(agent.tools))
    for goal in [
        "compute 17 * (9 + 4) - 3",
        "find quotes about love",
        "write a soliloquy",
        "word count of 'to be or not to be that is the question'",
    ]:
        print("\n>>>", goal)
        out = agent.run(goal)
        print(json.dumps(out, indent=2)[:1500])


if __name__ == "__main__":
    main()
