"""Phase 3 bridge: self-learning-llm-training -> model-onramp.

Agents resolve their model by ROLE through the on-ramp's capability router
instead of hard-coding model ids. Fallback is graceful at every layer:

  - model-onramp missing entirely  -> agents use their built-in defaults
  - no model probed for a role     -> that agent uses its built-in default
  - a model qualifies for the role -> the best-ranked (stable-first,
    cheapest) model serves, and a newly onboarded model takes over the
    moment it probes better — with zero changes in this repo.

Onboard models from the repo root with:

    PYTHONPATH=model-onramp python3 -m onramp probe --all
    PYTHONPATH=model-onramp python3 -m onramp promote <model-id>
"""

from __future__ import annotations

import sys
from pathlib import Path

_ONRAMP_DIR = Path(__file__).resolve().parent / "model-onramp"
if _ONRAMP_DIR.is_dir() and str(_ONRAMP_DIR) not in sys.path:
    sys.path.insert(0, str(_ONRAMP_DIR))

_router = None
_router_failed = False


def _get_router():
    global _router, _router_failed
    if _router is None and not _router_failed:
        try:
            from onramp.routing import RoleProfile, Router, load_roles

            roles = load_roles()
            # Roles specific to this repo's agent hierarchy, beyond the
            # on-ramp defaults (judge / meta_judge / trainer / ...).
            roles.setdefault("evaluator", RoleProfile(
                "evaluator", needs={"json_reliability": 0.8}))
            roles.setdefault("orchestrator", RoleProfile(
                "orchestrator", needs={"instruction_score": 0.8,
                                       "json_reliability": 0.8}))
            _router = Router(roles=roles)
        except Exception:
            _router_failed = True
    return _router


def resolve_model(role: str, default: str) -> str:
    """Best model for `role`, or `default` when the on-ramp can't help."""
    router = _get_router()
    if router is None or role not in router.roles:
        return default
    try:
        candidates = router.candidates(role)
    except Exception:
        return default
    return candidates[0] if candidates else default
