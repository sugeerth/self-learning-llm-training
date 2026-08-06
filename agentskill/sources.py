"""Ingest public agent trajectories into the common Trajectory schema.

Real sources (coding / ML / tool-use agents) plug in here; each is a function
that yields `Trajectory` objects. They require network/credentials, so in this
offline environment they degrade to an empty list and the pipeline falls back
to the synthetic generator — the seams are real, the data fetch is deferred.

To wire a real source: implement `<source>()` to download/parse its run logs
(e.g. SWE-bench agent trajectories from the leaderboard tarballs, GAIA run
dumps, ML-agent traces) and map each into a Trajectory. Keep the mapping in one
place so scoring/retrieval/eval stay source-agnostic.
"""

from __future__ import annotations

from .trajectory import Trajectory, synth_dataset

# name -> (description, hint on where the real data lives)
REGISTRY = {
    "swebench_agent": ("SWE-bench Verified agent run logs",
                       "https://www.swebench.com/ (per-model trajectory tarballs)"),
    "gaia_runs":      ("GAIA agent run dumps",
                       "GAIA leaderboard / HF dataset run exports"),
    "ml_agent":       ("ML-engineering agent traces",
                       "MLE-bench / open ML-agent trajectory dumps"),
}


def load_public(name: str, fetch=None) -> list[Trajectory]:
    """Load a named public source. `fetch(name) -> list[Trajectory]` can be
    injected (e.g. a downloader wired for Colab); without it, returns [] here
    so callers fall back to synthetic data instead of crashing offline."""
    if name not in REGISTRY:
        raise KeyError(f"unknown source '{name}'; known: {list(REGISTRY)}")
    if fetch is None:
        return []
    try:
        return list(fetch(name))
    except Exception:
        return []


def collect(names: list[str] | None = None, fetch=None,
            fallback_synthetic: bool = True, seed: int = 0) -> list[Trajectory]:
    """Collect from the named public sources; fall back to synthetic when the
    real fetch is unavailable (the common case in this sandbox)."""
    names = names or list(REGISTRY)
    out: list[Trajectory] = []
    for name in names:
        out.extend(load_public(name, fetch=fetch))
    if not out and fallback_synthetic:
        out = synth_dataset(seed=seed)
    return out
