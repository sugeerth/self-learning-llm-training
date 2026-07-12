"""CLI: python -m onramp <command>

  list                       registered models + manifest status
  probe <model-id>           run the capability probe suite (--all for every model)
  find --need k=v ...        capability query, cheapest first
  roles                      show role profiles
  resolve <role>             pick the best model for a role
  drift <model-id>           compare the two latest manifest snapshots
  history <model-id>         list manifest snapshots
  events [-n N]              tail the event stream
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .budget import BudgetExceededError
from .capabilities import CapabilityManifest, detect_drift
from .probes import run_probes
from .registry import get_registry
from .routing import NoEligibleModelError, Router
from . import events


def _parse_need(pairs: list[str]) -> dict:
    needs = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        needs[key] = (value.lower() == "true" if value.lower() in ("true", "false")
                      else float(value))
    return needs


def _probe_one(registry, model_id: str, args) -> None:
    print(f"probing {model_id} (budget ${args.budget:.2f}) ...")
    try:
        manifest = run_probes(registry.get(model_id), budget_usd=args.budget,
                              skip_context=args.skip_context)
    except BudgetExceededError as err:
        print(f"  stopped early: {err} — partial manifest saved")
        return
    print(json.dumps(asdict(manifest), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="onramp", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list")
    probe = sub.add_parser("probe")
    probe.add_argument("model_id", nargs="?")
    probe.add_argument("--all", action="store_true")
    probe.add_argument("--budget", type=float, default=1.0,
                       help="max USD to spend per model (default 1.0)")
    probe.add_argument("--skip-context", action="store_true",
                       help="skip the (most expensive) context probe")
    find = sub.add_parser("find")
    find.add_argument("--need", action="append", default=[],
                      metavar="KEY=VALUE", help="e.g. json_reliability=0.95")
    sub.add_parser("roles")
    resolve = sub.add_parser("resolve")
    resolve.add_argument("role")
    drift = sub.add_parser("drift")
    drift.add_argument("model_id")
    drift.add_argument("--threshold", type=float, default=0.10)
    history = sub.add_parser("history")
    history.add_argument("model_id")
    ev = sub.add_parser("events")
    ev.add_argument("-n", type=int, default=20)

    args = parser.parse_args()
    registry = get_registry()

    if args.command == "list":
        for model_id in registry.model_ids():
            manifest = registry.manifest(model_id)
            if manifest:
                print(f"{model_id:<28} json={manifest.json_reliability} "
                      f"tools={manifest.tool_use_reliability} "
                      f"instr={manifest.instruction_score} "
                      f"ctx={manifest.usable_context_tokens} "
                      f"tok/s={manifest.tokens_per_second} "
                      f"$out/M={manifest.output_per_mtok}")
            else:
                print(f"{model_id:<28} NOT PROBED  (python -m onramp probe {model_id})")
    elif args.command == "probe":
        targets = registry.model_ids() if args.all else [args.model_id]
        if not args.all and not args.model_id:
            parser.error("probe requires a model_id or --all")
        for model_id in targets:
            _probe_one(registry, model_id, args)
    elif args.command == "find":
        for model_id in registry.find(**_parse_need(args.need)):
            print(model_id)
    elif args.command == "roles":
        for role in Router().roles.values():
            print(f"{role.name:<12} prefer={role.prefer:<6} needs={role.needs}")
    elif args.command == "resolve":
        router = Router()
        try:
            best = router.resolve(args.role)
        except NoEligibleModelError as err:
            raise SystemExit(str(err))
        print(f"{args.role} -> {best}   (chain: {router.candidates(args.role)})")
    elif args.command == "drift":
        alerts = detect_drift(args.model_id, threshold=args.threshold)
        if alerts:
            print(f"DRIFT detected for {args.model_id}:")
            for alert in alerts:
                print(f"  {alert}")
            raise SystemExit(1)
        print(f"no drift (need >=2 snapshots; have "
              f"{len(CapabilityManifest.history(args.model_id))})")
    elif args.command == "history":
        for snap in CapabilityManifest.history(args.model_id):
            print(f"{snap.probed_at}  json={snap.json_reliability} "
                  f"tools={snap.tool_use_reliability} instr={snap.instruction_score} "
                  f"ctx={snap.usable_context_tokens} cost=${snap.probe_cost_usd}")
    elif args.command == "events":
        for event in events.tail(args.n):
            print(json.dumps(event))


if __name__ == "__main__":
    main()
