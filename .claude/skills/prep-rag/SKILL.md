---
name: prep-rag
description: >-
  Answer questions about THIS project by retrieving from its prep documents
  (README, PLAN, ARCHITECTURE, SELF_LEARNING, model-onramp docs) first, and the
  web only as a fallback. Use whenever the user asks how/why something in the
  repo works, where a concept is documented, wants a grounded/cited answer from
  the project's own docs, or wants to study/understand the codebase. Local-first,
  offline, no API key. Triggers: "how does X work here", "where is X documented",
  "explain the harness/onramp/agents", "quiz me on the docs".
---

# prep-rag — local-first retrieval over the project's documents

The prep documents are the source of truth. The web is only as useful as the
docs are thin — consult it **only** when the local corpus doesn't cover the
question, and treat anything it returns as secondary and lower-confidence.

## Answer a question (local-first, grounded, cited)

```sh
python -m rag query "how does the circuit breaker skip failing models"
```

Prints an extractive, cited answer with **coverage** (share of query terms the
prep docs cover) and **confidence**. `[prep]` passages come from the project's
documents; `[web]` passages (only if `--web` and coverage is low) are secondary.
Report the answer with its citations; if confidence is low or terms are listed
as missing, say so rather than guessing.

## When the docs may not cover it

```sh
python -m rag query --web "..."     # enables the secondary web source
```

The web source degrades to nothing on a blocked network (returns local-only).
To actually reach the web, inject a fetcher in code:
`rag.build_engine(root=".", web_fetcher=my_fetcher)` where `my_fetcher(query, k)`
returns `[(title, url, text), ...]`.

## Understand, don't just look up (interactive)

```sh
python -m rag study "circuit breaker and autopilot" -n 5   # active-recall Q&A
python -m rag clarify "how"                                 # is the query too vague?
python -m rag index                                        # list indexed chunks
```

`study` turns the retrieved passages into definitional and cloze (fill-in-the-
blank) questions — use it when the user wants to *learn* the material, not just
get an answer.

## In code

```python
from rag import build_engine
engine = build_engine(root=".")          # loads the prep documents
ans = engine.retrieve("what is the eval cache")
print(ans.grounded_text())               # cited synthesis, local-first
print(ans.coverage, ans.confidence, ans.used_web, ans.missing_terms)
```

## Rules

- Prefer `[prep]` passages; only fall back to `--web` when coverage is low, and
  label web-sourced claims as secondary.
- Always surface the citations (`doc_id` or URL) — never present a retrieved
  claim without its source.
- If `coverage` is low and no web is available, tell the user the prep docs
  don't cover it instead of fabricating an answer.
- It's offline and deterministic: no API key, safe to run anytime.
