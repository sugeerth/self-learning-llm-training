# rag — local-first retrieval over the project's prep documents

Answers questions by grounding in the project's **existing document universe**
(README, PLAN, ARCHITECTURE, SELF_LEARNING, model-onramp docs) first. The web
is strictly **secondary** — consulted only when the prep documents don't cover
the question, ranked below local results, and down-weighted in confidence.
Offline, deterministic, **no API key** (lexical BM25 retrieval), and it
degrades to local-only when the network is blocked.

## Why local-first

The prep documents are the source of truth for *this* project; the open web
isn't. So the engine only reaches for the web when the local corpus is thin,
and even then a web-reliant answer is capped in confidence and tagged
`treat as secondary`. A well-covered question never touches the web at all.

```
question
   │
   ▼
LOCAL prep docs ── BM25 ──▶ coverage = query terms found in top hits
   │                              │
   │                    coverage ≥ gate?  ── yes ──▶ answer (prep only)
   │                              │ no
   │                              ▼
   └────────────── web (gap terms only) ──▶ down-weight, rank below local
                                            ▼
                                  answer (cited, confidence tempered)
```

## CLI

```sh
python -m rag index                              # list indexed chunks
python -m rag query "how does the eval cache work"
python -m rag query --web "..."                  # allow the secondary web source
python -m rag study "circuit breaker" -n 5       # active-recall questions
python -m rag clarify "how"                      # is the query too vague?
```

## Library

```python
from rag import build_engine
engine = build_engine(root=".")                  # loads the prep documents
ans = engine.retrieve("what is autopilot")
print(ans.grounded_text())                       # cited, local-first synthesis
# ans.coverage, ans.confidence, ans.used_web, ans.missing_terms, ans.followups
```

To actually reach the web, pass a fetcher (`build_engine(root=".",
web_fetcher=fn)`) where `fn(query, k) -> [(title, url, text), ...]`.

## What's a "prep document"

Knowledge docs (`.md`, `.txt`) — **not** training corpora (`data/`), dependency
manifests, or caches, which are excluded so they can't pollute retrieval.

## Modules

| file | role |
|---|---|
| `corpus.py` | load + heading-aware chunk the prep documents |
| `retriever.py` | offline BM25 lexical index (deterministic) |
| `sources.py` | `LocalSource` (primary) + pluggable `WebSource` (secondary) |
| `engine.py` | local-first retrieval, coverage gate, grounding, study/clarify |

Also packaged as a Claude Code skill at `.claude/skills/prep-rag/`.
