"""CLI: python -m onramp <command>

  list                       registered models + manifest status
  probe <model-id>           run the capability probe suite
                             (--all for every model, --jobs N in parallel)
  discover [--probe]         auto-register new models from the Anthropic
                             Models API (zero adapter files)
  promote/demote/retire <id> lifecycle: candidate <-> stable, or retire
  autopilot [--apply]        promote/demote from live traffic evidence
  find --need k=v ...        capability query, best-ranked first
  roles                      show role profiles
  resolve <role>             pick the best model for a role
  drift <model-id>           compare the two latest manifest snapshots
  history <model-id>         list manifest snapshots
  export [--out FILE]        write the manifest feed JSON
  serve [--port N]           live dashboard (http://localhost:8010)
  events [-n N]              tail the event stream
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

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


def _probe_one(registry, model_id: str, args) -> str:
    try:
        manifest = run_probes(registry.get(model_id), budget_usd=args.budget,
                              skip_context=args.skip_context)
    except BudgetExceededError as err:
        return f"{model_id}: stopped early ({err}) — partial manifest saved"
    except Exception as err:
        return f"{model_id}: FAILED — {type(err).__name__}: {err}"
    return (f"{model_id}: json={manifest.json_reliability} "
            f"instr={manifest.instruction_score} tools={manifest.tool_use_reliability} "
            f"ctx={manifest.usable_context_tokens} cost=${manifest.probe_cost_usd}")


def _probe_many(registry, targets: list[str], args) -> None:
    print(f"probing {len(targets)} model(s), budget ${args.budget:.2f} each, "
          f"{args.jobs} parallel ...")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for line in pool.map(lambda m: _probe_one(registry, m, args), targets):
            print(" ", line)


def _set_status(model_id: str, status: str) -> None:
    manifest = CapabilityManifest.load(model_id)
    if manifest is None:
        raise SystemExit(f"{model_id} has no current manifest — probe it first")
    if status == "stable" and manifest.notes.get("pricing_unknown"):
        raise SystemExit(f"{model_id} has unknown pricing — set it in its "
                         f"adapter (or discovery.KNOWN_PRICING) before promoting")
    manifest.set_status(status)
    events.emit("status_change", model_id=model_id, status=status)
    print(f"{model_id} -> {status}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="onramp", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list")
    probe = sub.add_parser("probe")
    probe.add_argument("model_id", nargs="?")
    probe.add_argument("--all", action="store_true")
    probe.add_argument("--jobs", type=int, default=1)
    probe.add_argument("--budget", type=float, default=1.0,
                       help="max USD to spend per model (default 1.0)")
    probe.add_argument("--skip-context", action="store_true",
                       help="skip the (most expensive) context probe")
    disc = sub.add_parser("discover")
    disc.add_argument("--probe", action="store_true",
                      help="probe newly discovered models immediately")
    disc.add_argument("--jobs", type=int, default=2)
    disc.add_argument("--budget", type=float, default=1.0)
    disc.add_argument("--skip-context", action="store_true")
    for status_cmd in ("promote", "demote", "retire"):
        sub.add_parser(status_cmd).add_argument("model_id")
    auto = sub.add_parser("autopilot")
    auto.add_argument("--apply", action="store_true",
                      help="execute the actions (default: dry-run)")
    auto.add_argument("--min-calls", type=int, default=25)
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
    export = sub.add_parser("export")
    export.add_argument("--out", default="manifest-feed.json")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8010)
    ev = sub.add_parser("events")
    ev.add_argument("-n", type=int, default=20)

    args = parser.parse_args()
    registry = get_registry()

    if args.command == "list":
        for model_id in registry.model_ids():
            manifest = registry.manifest(model_id)
            if manifest:
                print(f"{model_id:<28} [{manifest.status:<9}] "
                      f"json={manifest.json_reliability} "
                      f"tools={manifest.tool_use_reliability} "
                      f"instr={manifest.instruction_score} "
                      f"ctx={manifest.usable_context_tokens} "
                      f"tok/s={manifest.tokens_per_second} "
                      f"$out/M={manifest.output_per_mtok}")
            else:
                print(f"{model_id:<28} NOT PROBED  (python -m onramp probe {model_id})")
    elif args.command == "probe":
        if not args.all and not args.model_id:
            parser.error("probe requires a model_id or --all")
        _probe_many(registry, registry.model_ids() if args.all else [args.model_id], args)
    elif args.command == "discover":
        from .discovery import discover

        new_ids = discover()
        if not new_ids:
            print("no new models — registry is current")
        for model_id in new_ids:
            print(f"discovered: {model_id}")
        if new_ids and args.probe:
            _probe_many(registry, new_ids, args)
    elif args.command in ("promote", "demote", "retire"):
        _set_status(args.model_id,
                    {"promote": "stable", "demote": "candidate",
                     "retire": "retired"}[args.command])
    elif args.command == "autopilot":
        from . import autopilot

        actions = autopilot.evaluate(min_calls=args.min_calls)
        if not actions:
            print("autopilot: no lifecycle changes supported by the evidence")
        for act in actions:
            print(f"{act.action.upper():<8} {act.model_id}  ({act.reason})")
        if actions and args.apply:
            autopilot.apply(actions)
            print(f"applied {len(actions)} action(s)")
        elif actions:
            print("dry-run — pass --apply to execute")
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
            print(f"{snap.probed_at}  [{snap.status}] json={snap.json_reliability} "
                  f"tools={snap.tool_use_reliability} instr={snap.instruction_score} "
                  f"ctx={snap.usable_context_tokens} cost=${snap.probe_cost_usd}")
    elif args.command == "export":
        from .dashboard import export_feed

        print(f"wrote {export_feed(args.out)}")
    elif args.command == "serve":
        from .dashboard import serve as run_server

        run_server(args.port)
    elif args.command == "events":
        for event in events.tail(args.n):
            print(json.dumps(event))


if __name__ == "__main__":
    main()
