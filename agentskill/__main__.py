"""CLI — the whole pipeline from the terminal (and from Colab).

  collect   [--out F]              gather trajectories (public sources or synth)
  score     [--in F] [--top N]     rank trajectories by learnable quality
  curate    [--in F] [--out F]     write the high-quality SFT dataset (jsonl)
  evaluate  [--json]               baseline vs trajectory-learned on the suite
  finetune  --sft F                LoRA fine-tune (needs a GPU; use on Colab)
"""

from __future__ import annotations

import argparse
import json

from .evaluate import compare, format_report
from .finetune import build_sft_dataset
from .scoring import rank
from .sources import collect
from .trajectory import load_jsonl, save_jsonl, synth_dataset


def main() -> None:
    ap = argparse.ArgumentParser(prog="agentskill", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect"); c.add_argument("--out", default="trajectories.jsonl")
    s = sub.add_parser("score"); s.add_argument("--in", dest="inp", default="trajectories.jsonl"); s.add_argument("--top", type=int, default=10)
    cu = sub.add_parser("curate"); cu.add_argument("--in", dest="inp", default="trajectories.jsonl"); cu.add_argument("--out", default="sft.jsonl"); cu.add_argument("--min-quality", type=float, default=0.6)
    e = sub.add_parser("evaluate"); e.add_argument("--json", action="store_true"); e.add_argument("--seed", type=int, default=0); e.add_argument("-k", type=int, default=5)
    f = sub.add_parser("finetune"); f.add_argument("--sft", default="sft.jsonl"); f.add_argument("--base-model", default="sshleifer/tiny-gpt2"); f.add_argument("--max-steps", type=int, default=60)

    args = ap.parse_args()

    if args.cmd == "collect":
        trajs = collect()
        save_jsonl(trajs, args.out)
        print(f"collected {len(trajs)} trajectories -> {args.out} "
              f"({sum(t.success for t in trajs)} successful)")
    elif args.cmd == "score":
        trajs = load_jsonl(args.inp) if _exists(args.inp) else synth_dataset()
        for t, sc in rank(trajs)[:args.top]:
            print(f"{sc.total:.3f}  {t.task_id:<22} success={t.success} "
                  f"recover={t.recovered}  {t.goal[:40]}")
    elif args.cmd == "curate":
        trajs = load_jsonl(args.inp) if _exists(args.inp) else synth_dataset()
        ex = build_sft_dataset(trajs, min_quality=args.min_quality, out_path=args.out)
        print(f"curated {len(ex)} SFT examples -> {args.out}")
    elif args.cmd == "evaluate":
        result = compare(seed=args.seed, k=args.k)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(format_report(result))
    elif args.cmd == "finetune":
        from .finetune import lora_finetune
        out = lora_finetune(args.sft, base_model=args.base_model,
                            max_steps=args.max_steps)
        print(f"LoRA adapter saved -> {out}")


def _exists(p: str) -> bool:
    import os
    return os.path.exists(p)


if __name__ == "__main__":
    main()
