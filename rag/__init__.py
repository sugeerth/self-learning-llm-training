"""Local-first RAG over the project's prep documents.

Grounds answers in the existing document universe first; the web is secondary,
consulted only when the prep docs don't cover the question and always
down-weighted. Offline and deterministic — no API key, no network required.
"""

from .corpus import Document, load_corpus
from .engine import Answer, RagEngine, build_engine
from .retriever import BM25Index, Hit, content_terms, tokenize
from .sources import LocalSource, WebSource

__all__ = [
    "Answer", "BM25Index", "Document", "Hit", "LocalSource", "RagEngine",
    "WebSource", "build_engine", "content_terms", "load_corpus", "tokenize",
]
