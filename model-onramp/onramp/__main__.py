"""CLI: python -m onramp list | probe <model-id>"""

import argparse
import json
from dataclasses import asdict

from .probes import run_probes
from .registry import get_registry


def main() -> None:
    parser = argparse.ArgumentParser(prog="onramp")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list registered models and manifest status")

    probe = sub.add_parser("probe", help="run capability probes on a model")
    probe.add_argument("model_id")

    args = parser.parse_args()
    registry = get_registry()

    if args.command == "list":
        for model_id in registry.model_ids():
            manifest = registry.manifest(model_id)
            status = "probed" if manifest else "NOT PROBED (run: onramp probe)"
            print(f"{model_id:<30} {status}")
            if manifest:
                print(f"{'':<30} json={manifest.json_reliability} "
                      f"tok/s={manifest.tokens_per_second} "
                      f"$out/M={manifest.output_per_mtok}")
    elif args.command == "probe":
        model = registry.get(args.model_id)
        manifest = run_probes(model)
        print(json.dumps(asdict(manifest), indent=2))
        print(f"\nmanifest written to {manifest.path}")


if __name__ == "__main__":
    main()
