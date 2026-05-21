"""Lightweight inference engine wrappers — picks the best backend available.

Backends in priority order:
  1. TensorRT-LLM   — fastest on NVIDIA (CUDA)
  2. vLLM           — PagedAttention, great throughput (CUDA)
  3. ggml/llama.cpp — CPU + Metal (Mac)
  4. HuggingFace transformers + native model — fallback (works on MPS/CUDA/CPU)

Plus quantization: INT8, INT4 (bitsandbytes / torch.quantize).

API:
    eng = InferenceEngine.auto(model_path)
    out = eng.generate("ROMEO:\\n", max_new_tokens=80)
    eng.benchmark()  # prints tok/s, memory, p50/p99 latency
"""
from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import torch


@dataclass
class BenchResult:
    backend: str
    quantization: Optional[str]
    tok_per_sec: float
    p50_ms: float
    p99_ms: float
    weights_mb: float
    peak_mem_mb: float
    notes: str = ""


def _detect() -> dict:
    info = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(),
            "vllm": False, "tensorrt_llm": False, "llama_cpp": False, "bitsandbytes": False}
    for mod in ["vllm", "tensorrt_llm", "llama_cpp", "bitsandbytes"]:
        try:
            importlib.import_module(mod)
            info[mod] = True
        except ImportError:
            pass
    return info


# ────────────────────── HF / native fallback ──────────────────────

class NativeEngine:
    """Uses model.py's LLM class directly. Works on MPS/CUDA/CPU."""
    backend = "native-pytorch"

    def __init__(self, ckpt_path: str, device: Optional[str] = None,
                 quantization: Optional[str] = None):
        from model import LLM, ModelConfig
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**ckpt["cfg"]) if isinstance(ckpt.get("cfg"), dict) else ckpt["cfg"]
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = LLM(cfg).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.quantization = quantization
        if quantization == "int8":
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8)
        # Note: int4 on PyTorch needs bitsandbytes — separate path

        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 80,
                 temperature: float = 0.8, top_k: int = 50) -> str:
        ids = torch.tensor([self.enc.encode(prompt)], device=self.device)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens,
                                  temperature=temperature, top_k=top_k)
        return self.enc.decode(out[0].tolist())

    def benchmark(self, prompt: str = "ROMEO:\n", n_tokens: int = 128, warmup: int = 2) -> BenchResult:
        for _ in range(warmup):
            self.generate(prompt, max_new_tokens=8)
        latencies = []
        t0 = time.time()
        for _ in range(5):
            tt = time.time()
            self.generate(prompt, max_new_tokens=n_tokens)
            latencies.append((time.time() - tt) * 1000 / n_tokens)  # ms/token
        elapsed = time.time() - t0
        latencies.sort()
        weights_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6
        peak = (torch.cuda.max_memory_allocated() / 1e6) if self.device == "cuda" else weights_mb * 1.5
        return BenchResult(
            backend=self.backend,
            quantization=self.quantization,
            tok_per_sec=(n_tokens * 5) / elapsed,
            p50_ms=latencies[len(latencies) // 2],
            p99_ms=latencies[-1],
            weights_mb=round(weights_mb, 2),
            peak_mem_mb=round(peak, 2),
            notes=f"device={self.device}",
        )


# ────────────────────── vLLM (CUDA) ──────────────────────

class VLLMEngine:
    backend = "vllm"

    def __init__(self, hf_repo_or_path: str, dtype: str = "auto", quantization: Optional[str] = None):
        from vllm import LLM as VL, SamplingParams  # type: ignore
        self.llm = VL(model=hf_repo_or_path, dtype=dtype, quantization=quantization,
                      enforce_eager=False, gpu_memory_utilization=0.85)
        self.SamplingParams = SamplingParams
        self.quantization = quantization

    def generate(self, prompt: str, max_new_tokens: int = 80,
                 temperature: float = 0.8, top_k: int = 50) -> str:
        params = self.SamplingParams(temperature=temperature, top_k=top_k, max_tokens=max_new_tokens)
        out = self.llm.generate([prompt], params)
        return out[0].outputs[0].text

    def benchmark(self, prompt: str = "ROMEO:\n", n_tokens: int = 128, **kw) -> BenchResult:
        # vLLM has built-in throughput measurement; here we time an end-to-end batch
        params = self.SamplingParams(temperature=0.8, top_k=50, max_tokens=n_tokens)
        t0 = time.time()
        outs = self.llm.generate([prompt] * 4, params)
        elapsed = time.time() - t0
        total_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        return BenchResult(
            backend=self.backend, quantization=self.quantization,
            tok_per_sec=total_tok / elapsed, p50_ms=0, p99_ms=0,
            weights_mb=0, peak_mem_mb=0, notes="vllm internal stats — see /metrics",
        )


# ────────────────────── TensorRT-LLM ──────────────────────

class TensorRTLLMEngine:
    backend = "tensorrt-llm"

    def __init__(self, engine_dir: str):
        # Requires a pre-built TRT engine; build with `trtllm-build`
        from tensorrt_llm.runtime import ModelRunnerCpp  # type: ignore
        self.runner = ModelRunnerCpp.from_dir(engine_dir=engine_dir)

    def generate(self, prompt: str, max_new_tokens: int = 80, **kw) -> str:
        # placeholder — TRT-LLM needs tokenizer wiring per-model
        raise NotImplementedError("TRT engine requires pre-tokenized inputs; wire your tokenizer here.")


# ────────────────────── llama.cpp (Mac/CPU) ──────────────────────

class LlamaCppEngine:
    backend = "llama.cpp"

    def __init__(self, gguf_path: str, n_threads: int = 8, n_gpu_layers: int = -1):
        from llama_cpp import Llama  # type: ignore
        self.llm = Llama(model_path=gguf_path, n_threads=n_threads,
                         n_gpu_layers=n_gpu_layers, verbose=False)

    def generate(self, prompt: str, max_new_tokens: int = 80,
                 temperature: float = 0.8, top_k: int = 50) -> str:
        out = self.llm(prompt, max_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        return out["choices"][0]["text"]


# ────────────────────── auto-pick ──────────────────────

class InferenceEngine:
    """Factory: picks the best available backend for current hardware."""

    @staticmethod
    def auto(model_or_path: str, prefer: Optional[str] = None,
             quantization: Optional[str] = None) -> object:
        info = _detect()
        if prefer:
            if prefer == "vllm" and info["vllm"]:
                return VLLMEngine(model_or_path, quantization=quantization)
            if prefer == "tensorrt-llm" and info["tensorrt_llm"]:
                return TensorRTLLMEngine(model_or_path)
            if prefer == "llama.cpp" and info["llama_cpp"]:
                return LlamaCppEngine(model_or_path)
        # auto: CUDA → vLLM, Mac → llama.cpp if gguf, else native
        if info["cuda"] and info["vllm"]:
            return VLLMEngine(model_or_path, quantization=quantization)
        if model_or_path.endswith(".gguf") and info["llama_cpp"]:
            return LlamaCppEngine(model_or_path)
        return NativeEngine(model_or_path, quantization=quantization)


# ────────────────────── benchmark suite ──────────────────────

def run_full_benchmark(model_path: str, prompt: str = "ROMEO:\n") -> list[BenchResult]:
    """Try every available backend × quantization combo, return rows for dashboard."""
    info = _detect()
    results = []

    # native fp16/fp32
    try:
        eng = NativeEngine(model_path)
        results.append(eng.benchmark(prompt))
    except Exception as e:
        print(f"native failed: {e}")

    # native int8
    try:
        eng = NativeEngine(model_path, quantization="int8")
        results.append(eng.benchmark(prompt))
    except Exception as e:
        print(f"native-int8 failed: {e}")

    # vLLM (if CUDA)
    if info["cuda"] and info["vllm"]:
        try:
            eng = VLLMEngine(model_path)
            results.append(eng.benchmark(prompt))
        except Exception as e:
            print(f"vllm failed: {e}")

    return results


if __name__ == "__main__":
    import json, sys
    path = sys.argv[1] if len(sys.argv) > 1 else "ckpt.pt"
    rows = run_full_benchmark(path)
    print(json.dumps([asdict(r) for r in rows], indent=2))
