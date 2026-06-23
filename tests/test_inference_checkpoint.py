"""Regression test for the NativeEngine checkpoint-key bug.

Checkpoints in this repo are saved as {"model": state_dict, "cfg": ...}
(see train.py / experiments.py). NativeEngine previously read ckpt["state_dict"],
a key that is never written, so it crashed with KeyError on every real checkpoint.
"""
from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")


def _save_tiny_ckpt(path: str):
    from model import LLM, ModelConfig

    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=2,
                      n_heads=2, n_kv_heads=1, max_seq_len=16)
    model = LLM(cfg)
    # exactly the format train.py / experiments.py write
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, path)


def test_native_engine_loads_model_key_checkpoint():
    from inference import NativeEngine

    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        _save_tiny_ckpt(path)
        eng = NativeEngine(path, device="cpu")  # must not raise KeyError
        assert eng.model is not None
        assert eng.device == "cpu"
    finally:
        os.remove(path)


def test_native_engine_still_accepts_legacy_state_dict_key():
    """Forward-compat: a checkpoint saved under the legacy 'state_dict' key still loads."""
    from inference import NativeEngine
    from model import LLM, ModelConfig

    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=2,
                      n_heads=2, n_kv_heads=1, max_seq_len=16)
    model = LLM(cfg)

    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, path)
        eng = NativeEngine(path, device="cpu")
        assert eng.model is not None
    finally:
        os.remove(path)
