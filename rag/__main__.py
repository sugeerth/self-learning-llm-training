"""CLI: python -m rag <command>

  index                      list the prep-document chunks that were loaded
  query "<question>"         local-first grounded answer (--web enables the
                             secondary web source; blocked networks degrade
                             to local-only automatically)
  study "<topic>" [-n N]     active-recall questions generated from the docs
  clarify "<question>"       interactive: is the query too vague/broad?
"""

from __future__ import annotations

import argparse
import json

from .engine import build_engine


def _demo_web_fetcher(query: str, k: int):
    """Placeholder secondary source for --web when no real fetcher is wired.
    Returns nothing (the container blocks outbound HTTP), which exercises the
    graceful-degradation path: the engine falls back to local-only."""
    return []


def main() -> None:
    # shared flags live on a parent so they parse before OR after the command
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=".", help="corpus root (default: repo)")
    common.add_argument("--web", action="store_true",
                        help="enable the secondary web source (degrades if blocked)")
    common.add_argument("--gate", type=float, default=0.6,
                        help="coverage below this consults the web (default 0.6)")

    ap = argparse.ArgumentParser(prog="rag", description=__doc__, parents=[common],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("index", parents=[common])
    q = sub.add_parser("query", parents=[common]); q.add_argument("question")
    s = sub.add_parser("study", parents=[common]); s.add_argument("topic"); s.add_argument("-n", type=int, default=5)
    c = sub.add_parser("clarify", parents=[common]); c.add_argument("question")

    args = ap.parse_args()
    engine = build_engine(root=args.root,
                          web_fetcher=_demo_web_fetcher if args.web else None,
                          coverage_gate=args.gate)

    if args.cmd == "index":
        docs = engine.local.docs
        print(f"{len(docs)} chunks from "
              f"{len({d.source for d in docs})} prep documents:")
        for d in docs:
            print(f"  {d.doc_id}  ({len(d.text)} chars)")
    elif args.cmd == "query":
        clar = engine.clarify(args.question)
        if clar:
            print("clarify:")
            for c in clar:
                print("  -", c)
            print()
        ans = engine.retrieve(args.question)
        print(ans.grounded_text())
        if ans.followups:
            print("\nfollow-ups:")
            for f in ans.followups:
                print("  -", f)
    elif args.cmd == "study":
        for i, item in enumerate(engine.study_questions(args.topic, n=args.n), 1):
            print(f"{i}. [{item['type']}] {item['q']}")
            if "answer" in item:
                print(f"   answer: {item['answer']}")
            print(f"   ({item['source']})")
    elif args.cmd == "clarify":
        cs = engine.clarify(args.question)
        print(json.dumps(cs, indent=2) if cs else "Query looks specific enough.")


if __name__ == "__main__":
    main()
