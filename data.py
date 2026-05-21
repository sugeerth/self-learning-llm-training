"""Data loading. Uses Tiny Shakespeare (1MB, trains fast on a laptop).

Tokenizer: GPT-2 BPE via tiktoken (50257 vocab). We pad to 50304 for speed.
Dataset is pre-tokenized to disk as uint16 bin files (nanoGPT convention).
"""
from __future__ import annotations

import os
import ssl
import subprocess
import urllib.request

import numpy as np
import tiktoken
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare() -> tuple[str, str]:
    os.makedirs(DATA_DIR, exist_ok=True)
    txt_path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    train_bin = os.path.join(DATA_DIR, "train.bin")
    val_bin = os.path.join(DATA_DIR, "val.bin")

    if not os.path.exists(txt_path):
        print(f"Downloading tiny shakespeare → {txt_path}")
        try:
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(URL, context=ctx, timeout=20) as r, open(txt_path, "wb") as f:
                f.write(r.read())
        except Exception:
            # Fallback to curl (uses system CAs, works when Python's bundle is stale)
            subprocess.run(["curl", "-fsSL", URL, "-o", txt_path], check=True)

    if not (os.path.exists(train_bin) and os.path.exists(val_bin)):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        ids = np.array(enc.encode_ordinary(text), dtype=np.uint16)
        n = int(0.9 * len(ids))
        ids[:n].tofile(train_bin)
        ids[n:].tofile(val_bin)
        print(f"Tokens: train={n:,} val={len(ids) - n:,}")
    return train_bin, val_bin


class Loader:
    def __init__(self, bin_path: str, block_size: int, batch_size: int, device: str):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block = block_size
        self.bs = batch_size
        self.device = device

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data) - self.block - 1, (self.bs,))
        x = torch.stack([torch.from_numpy(self.data[i : i + self.block].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + self.block].astype(np.int64)) for i in ix])
        # non_blocking is a no-op on MPS but free on CUDA
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def tokenizer():
    return tiktoken.get_encoding("gpt2")
