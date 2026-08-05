"""Offline lexical retrieval — BM25 over the document chunks.

Pure Python (stdlib only): no embedding API, no network, no torch. Retrieval
is a deterministic function of the corpus + query, which is exactly what a
no-API-key, reproducible pipeline needs. BM25 (Okapi) is a strong lexical
baseline that rewards rare query terms and is length-normalized, so a short
precise section isn't buried under a long rambling one.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .corpus import Document

_STOP = set("""a an the of to in on for and or but is are was were be been being
this that these those it its as at by from with without into out up down over
under again further then once here there all any both each few more most other
some such no nor not only own same so than too very can will just don should now
i you he she we they them his her their our your my me us do does did has have
had how what when where which who whom why with about above below""".split())

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords dropped, 1-char tokens dropped."""
    return [t for t in _TOKEN.findall(text.lower())
            if t not in _STOP and len(t) > 1]


def content_terms(query: str) -> list[str]:
    """Distinct content tokens of a query — the terms we expect coverage of."""
    seen, out = set(), []
    for t in tokenize(query):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


@dataclass
class Hit:
    doc: Document
    score: float
    matched_terms: list[str]


class BM25Index:
    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.toks = [tokenize(d.text + " " + d.title) for d in docs]
        self.len = [len(t) for t in self.toks]
        self.avglen = (sum(self.len) / len(self.len)) if self.len else 0.0
        self.tf = [Counter(t) for t in self.toks]
        df: Counter = Counter()
        for t in self.toks:
            df.update(set(t))
        n = max(len(docs), 1)
        # BM25 idf with the +1 inside the log so it's always positive
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5))
                    for term, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = content_terms(query)
        hits: list[Hit] = []
        for i, doc in enumerate(self.docs):
            score, matched = 0.0, []
            dl = self.len[i] or 1
            for term in q:
                f = self.tf[i].get(term, 0)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avglen or 1))
                score += idf * (f * (self.k1 + 1)) / denom
                matched.append(term)
            if score > 0:
                hits.append(Hit(doc=doc, score=score, matched_terms=matched))
        hits.sort(key=lambda h: (-h.score, h.doc.doc_id))   # deterministic ties
        return hits[:k]
