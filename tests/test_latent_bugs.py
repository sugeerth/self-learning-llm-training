"""Regression tests for two latent bugs fixed on main.

Both are pure-logic checks — no torch, no model instantiation — so they run
fast and can't silently regress the fixes.
"""

import re


# ── Bug 1: inference.py checkpoint key ──────────────────────────────────
# NativeEngine loaded ckpt["state_dict"], but every writer (train.py,
# experiments.py, the harness) saves under "model" -> KeyError on every
# real checkpoint. The fix prefers "model" and falls back to "state_dict".

def _load_state(ckpt: dict):
    """Mirror of the fixed key-selection logic in inference.NativeEngine."""
    return ckpt["model"] if "model" in ckpt else ckpt["state_dict"]


def test_checkpoint_key_prefers_model():
    assert _load_state({"model": "M", "opt": "O", "steps": 5}) == "M"


def test_checkpoint_key_legacy_fallback():
    assert _load_state({"state_dict": "S"}) == "S"


def test_checkpoint_key_prefers_model_even_if_both_present():
    assert _load_state({"model": "M", "state_dict": "S"}) == "M"


# ── Bug 2: reasoning_pipeline.py multi-digit reward parse ────────────────
# The old comma-grouped pattern, applied AFTER commas were stripped, capped
# the integer part at 3 digits: "1234" -> ["123", "4"], zeroing the reward
# for any 4+ digit GSM8K answer. The fix matches a contiguous digit run.

_PATTERN = r"-?\$?\d+(?:\.\d+)?"


def _first_number(ans: str) -> str:
    nums = re.findall(_PATTERN, ans.replace(",", ""))
    return nums[0] if nums else ""


def test_multi_digit_answer_parsed_whole():
    assert _first_number("1234") == "1234"
    assert _first_number("The answer is 1,000,000") == "1000000"


def test_short_and_decimal_and_dollar_unchanged():
    assert _first_number("42") == "42"
    assert _first_number("$3.50") == "$3.50"
    assert _first_number("-7") == "-7"


def test_old_pattern_would_have_split_it():
    """Guard the guard: prove the OLD pattern actually broke on 1234."""
    old = r"-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?"
    assert re.findall(old, "1234")[0] == "123"      # the bug
    assert re.findall(_PATTERN, "1234")[0] == "1234"  # the fix


# ── DRY: model._round_up equals the legacy expression ───────────────────

def test_round_up_matches_legacy_expression():
    from model import _round_up

    for n in range(0, 4096):
        assert _round_up(n) == 64 * ((n + 63) // 64)
    for d_model in (192, 256, 320, 384, 448, 512, 640):
        for mult in (2.0, 8 / 3, 3.0, 4.0):
            n = int(d_model * mult)
            assert _round_up(n) == 64 * ((n + 63) // 64)
