"""RagEngine — local-first retrieval + grounded understanding.

Design principle (the user's ask): the prep documents are what matter. The web
is only as useful as the local corpus is thin. So:

  1. Retrieve from the LOCAL prep documents first.
  2. Compute COVERAGE = fraction of the query's content terms found in the top
     local hits.
  3. Only if coverage is below `coverage_gate` do we consult the web, and only
     to fill the *uncovered* terms (gap-aware). Web hits are down-weighted by
     WEB_PENALTY and always ranked below local hits.
  4. Answers are grounded: every passage cites its source, provenance (local
     vs web) is explicit, confidence reflects coverage, and gaps are named.

Understanding-first, interactive techniques (all offline, deterministic):
  - query expansion (full query + content-only + per-gap re-queries)
  - gap-aware multi-hop retrieval
  - clarify(): flags vague/ambiguous queries and proposes sharper ones
  - study_questions(): turns retrieved docs into active-recall questions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retriever import Hit, content_terms
from .sources import LocalSource, Source, WebSource

WEB_PENALTY = 0.35   # web hits keep only this share of their score vs local


@dataclass
class Answer:
    query: str
    passages: list[Hit]
    coverage: float                       # 0..1 local coverage of query terms
    confidence: float                     # 0..1 overall
    used_web: bool
    covered_terms: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)

    def grounded_text(self, max_passages: int = 3) -> str:
        """A readable, cited synthesis — local passages first."""
        if not self.passages:
            return ("No prep document covers this. "
                    + ("The web source is unavailable too."
                       if not self.used_web else "The web had nothing either."))
        lines = []
        for h in self.passages[:max_passages]:
            tag = "web" if h.doc.origin == "web" else "prep"
            snippet = _best_sentences(h.doc.text, h.matched_terms)
            cite = h.doc.url or h.doc.doc_id
            lines.append(f"[{tag}] {snippet}\n    — {cite}")
        head = (f"Grounded in {len(self.passages)} passage(s); "
                f"coverage {self.coverage:.0%}, confidence {self.confidence:.0%}"
                + (" (used web — treat as secondary)" if self.used_web else
                   " (prep documents only)") + ".")
        return head + "\n\n" + "\n\n".join(lines)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]


def _best_sentences(text: str, terms: list[str], n: int = 2) -> str:
    """Extractive snippet: the sentences densest in the matched terms."""
    sents = _sentences(text)
    if not sents:
        return text[:240]
    tset = set(terms)
    scored = sorted(sents, key=lambda s: -sum(
        t in s.lower() for t in tset))[:n]
    # restore original order for readability
    inorder = [s for s in sents if s in scored]
    return " ".join(inorder)[:400]


class RagEngine:
    def __init__(self, local: LocalSource, web: WebSource | None = None,
                 coverage_gate: float = 0.6, k: int = 5):
        self.local = local
        self.web = web
        self.coverage_gate = coverage_gate
        self.k = k

    # -- query expansion -------------------------------------------------

    def expand(self, query: str) -> list[str]:
        """Full query + a content-terms-only variant. Deterministic, offline."""
        terms = content_terms(query)
        variants = [query]
        cq = " ".join(terms)
        if cq and cq != query.lower():
            variants.append(cq)
        return variants

    def _coverage(self, query: str, hits: list[Hit]):
        terms = content_terms(query)
        if not terms:
            return 1.0, [], []
        found = set()
        for h in hits:
            found.update(h.matched_terms)
        covered = [t for t in terms if t in found]
        missing = [t for t in terms if t not in found]
        return len(covered) / len(terms), covered, missing

    # -- retrieval -------------------------------------------------------

    def retrieve(self, query: str) -> Answer:
        # 1. local-first, across expanded queries, merged by best score
        local_hits = self._merge(
            [h for v in self.expand(query) for h in self.local.search(v, self.k)])
        coverage, covered, missing = self._coverage(query, local_hits)

        used_web = False
        passages = list(local_hits[:self.k])
        # 2. coverage gate: only reach for web if the prep docs fall short
        if self.web and missing and coverage < self.coverage_gate:
            gap_query = " ".join(missing) or query
            web_hits = self.web.search(gap_query, self.k)
            if web_hits:
                used_web = True
                for h in web_hits:            # 3. down-weight + rank below local
                    h.score *= WEB_PENALTY
                passages = self._merge(local_hits + web_hits)
                # recompute coverage now that web filled gaps
                coverage, covered, missing = self._coverage(query, passages)

        # 4. confidence: coverage, tempered when the answer leans on web
        conf = coverage * (0.75 if used_web else 1.0)
        if not passages:
            conf = 0.0
        return Answer(
            query=query, passages=passages[:self.k], coverage=coverage,
            confidence=round(conf, 3), used_web=used_web,
            covered_terms=covered, missing_terms=missing,
            followups=self._followups(query, missing, passages))

    def _merge(self, hits: list[Hit]) -> list[Hit]:
        """Dedupe by doc_id keeping the best score; local outranks web on ties."""
        best: dict[str, Hit] = {}
        for h in hits:
            cur = best.get(h.doc.doc_id)
            if cur is None or h.score > cur.score:
                best[h.doc.doc_id] = h
        merged = list(best.values())
        merged.sort(key=lambda h: (h.doc.origin == "web", -h.score, h.doc.doc_id))
        return merged

    # -- interactive understanding --------------------------------------

    def clarify(self, query: str) -> list[str]:
        """Flag a vague query and propose sharper ones — before retrieving."""
        terms = content_terms(query)
        qs: list[str] = []
        if len(terms) <= 1:
            qs.append("Your query is very broad — which specific aspect? "
                      "(e.g. how it works, why it exists, how to run it)")
        probe = self.local.search(" ".join(terms) or query, k=6)
        distinct_sources = {h.doc.source for h in probe}
        if len(distinct_sources) >= 4:
            qs.append("This spans several documents "
                      f"({', '.join(sorted(distinct_sources)[:4])}…) — "
                      "narrow to one to go deeper?")
        return qs

    def _followups(self, query, missing, passages) -> list[str]:
        fu = []
        if missing:
            fu.append("The prep docs don't cover: " + ", ".join(missing[:5]))
        # suggest adjacent sections the reader might want next
        for h in passages[:2]:
            if h.doc.origin == "local":
                fu.append(f"Related section: {h.doc.title} ({h.doc.source})")
        return fu[:4]

    def study_questions(self, topic: str, n: int = 5) -> list[dict]:
        """Creative active-recall: turn the top retrieved prep passages into
        questions (definitional from headings, cloze from key sentences), so
        the corpus becomes a study aid, not just a lookup table."""
        hits = self.local.search(topic, k=max(n, 3))
        out: list[dict] = []
        for h in hits:
            # definitional question from the section heading
            if h.doc.title and not h.doc.title.endswith(".md"):
                out.append({"q": f"What does \"{h.doc.title}\" mean / cover?",
                            "type": "definitional", "source": h.doc.doc_id})
            # cloze from the sentence densest in matched terms
            snip = _best_sentences(h.doc.text, h.matched_terms, n=1)
            if snip and h.matched_terms:
                blank = h.matched_terms[0]
                cloze = re.sub(rf"\b{re.escape(blank)}\b", "____", snip,
                               count=1, flags=re.IGNORECASE)
                if "____" in cloze:
                    out.append({"q": cloze, "answer": blank, "type": "cloze",
                                "source": h.doc.doc_id})
            if len(out) >= n:
                break
        return out[:n]


def build_engine(root: str = ".", web_fetcher=None, **kw) -> RagEngine:
    """Convenience: load the prep documents and wire the engine."""
    from .corpus import load_corpus
    local = LocalSource(load_corpus(root))
    web = WebSource(web_fetcher) if web_fetcher is not None else None
    return RagEngine(local, web, **kw)
