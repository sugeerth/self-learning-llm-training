"""Trajectory memory + retrieval — "retrieve relevant past experiences".

Reuses the repo's offline BM25 retriever (rag/) so a new task can pull the most
similar high-quality past trajectories. Falls back to a tiny built-in lexical
scorer if the rag package isn't importable, so agentskill stands alone.
"""

from __future__ import annotations

from .scoring import QualityScore, curate
from .trajectory import Trajectory

try:                                   # reuse the project's BM25 if present
    from rag.retriever import BM25Index
    from rag.corpus import Document
    _HAVE_RAG = True
except Exception:                      # pragma: no cover - standalone fallback
    _HAVE_RAG = False


def _fallback_search(query: str, entries, k):
    q = set(w for w in query.lower().split() if len(w) > 1)
    scored = []
    for traj, score in entries:
        toks = set(traj.text().lower().split())
        overlap = len(q & toks)
        if overlap:
            scored.append((overlap, traj, score))
    scored.sort(key=lambda x: (-x[0], x[1].task_id))
    return [(t, s) for _, t, s in scored[:k]]


class TrajectoryMemory:
    """Holds curated high-quality trajectories and retrieves by task similarity.
    Quality-gated at build time so retrieval only ever surfaces good exemplars
    (the "learn from successful experiences" half); the raw pool still keeps
    failures for contrastive/analysis use."""

    def __init__(self, trajs: list[Trajectory], min_quality: float = 0.6):
        self.raw = trajs
        self.entries: list[tuple[Trajectory, QualityScore]] = curate(
            trajs, min_quality=min_quality)
        self._index = None
        if _HAVE_RAG and self.entries:
            docs = [Document(doc_id=t.task_id, source=t.source, title=t.goal,
                             text=t.text()) for t, _ in self.entries]
            self._index = BM25Index(docs)
            self._by_id = {t.task_id: (t, s) for t, s in self.entries}

    def __len__(self) -> int:
        return len(self.entries)

    def retrieve(self, query: str, k: int = 5
                 ) -> list[tuple[Trajectory, QualityScore]]:
        if not self.entries:
            return []
        if self._index is not None:
            hits = self._index.search(query, k=k)
            return [self._by_id[h.doc.doc_id] for h in hits
                    if h.doc.doc_id in self._by_id]
        return _fallback_search(query, self.entries, k)
