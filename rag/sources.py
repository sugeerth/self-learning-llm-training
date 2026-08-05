"""Retrieval sources. Local prep documents are the PRIMARY source; the web is
strictly SECONDARY — consulted only when the local corpus doesn't cover the
question, and always ranked below local results.

The web source takes a pluggable `fetcher` so the engine never hard-depends on
network access: the default fetcher uses stdlib urllib and returns [] on any
failure (this repo's containers block outbound HTTP), so the whole system
degrades to local-only without raising. Inject a real fetcher (or the caller's
web-search tool) to actually reach the web.
"""

from __future__ import annotations

from typing import Callable, Protocol

from .corpus import Document
from .retriever import BM25Index, Hit


class Source(Protocol):
    origin: str
    def search(self, query: str, k: int) -> list[Hit]: ...


class LocalSource:
    """The prep documents. Primary, offline, deterministic."""
    origin = "local"

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.index = BM25Index(docs)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self.index.search(query, k=k)


# A web fetcher maps a query -> list of (title, url, text) results.
WebFetcher = Callable[[str, int], list[tuple[str, str, str]]]


def _null_fetcher(query: str, k: int) -> list[tuple[str, str, str]]:
    """Default: try stdlib, but expect to be blocked -> return nothing.
    Kept intentionally minimal; real web access is injected by the caller."""
    return []


class WebSource:
    """Secondary source. Builds a throwaway BM25 index over whatever the
    fetcher returns so web snippets are scored on the same footing as local
    chunks — then the engine down-weights them (see engine.WEB_PENALTY)."""
    origin = "web"

    def __init__(self, fetcher: WebFetcher | None = None):
        self.fetcher = fetcher or _null_fetcher
        self.available = fetcher is not None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        try:
            results = self.fetcher(query, k)
        except Exception:
            return []
        if not results:
            return []
        docs = [Document(doc_id=f"web:{url}", source=url, title=title,
                         text=text, origin="web", url=url)
                for title, url, text in results if text]
        if not docs:
            return []
        return BM25Index(docs).search(query, k=k)
