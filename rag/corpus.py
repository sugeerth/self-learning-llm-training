"""The document universe: load and chunk the local 'prep documents'.

A Document is one retrievable chunk with a stable id and provenance back to
its source file, so every answer can cite exactly where it came from. Markdown
is chunked by heading section (understanding lives in sections, not arbitrary
token windows); plain text falls back to paragraph blocks.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

# Repo docs that make up the default "prep documents" universe.
DEFAULT_GLOBS = ["*.md", "*.txt", "**/*.md", "**/*.txt"]
# The prep documents are KNOWLEDGE docs — not the training corpora under data/
# (e.g. tinyshakespeare.txt is model input, not a document to answer from),
# nor dependency manifests, nor build/cache dirs.
EXCLUDE_DIRS = (".git", ".pytest_cache", "node_modules", ".onramp", "runs",
                "data", "__pycache__")
EXCLUDE_FILES = ("requirements.txt",)


@dataclass
class Document:
    doc_id: str          # "README.md#the-idea"
    source: str          # relative file path
    title: str           # section heading (or file name)
    text: str
    origin: str = "local"   # "local" (prep docs) or "web"
    url: str | None = None  # set for web-origin documents
    meta: dict = field(default_factory=dict)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "section"


def _chunk_markdown(path: str, text: str, max_chars: int = 1400) -> list[Document]:
    """Split on ATX headings; further split long sections on blank lines so no
    chunk dwarfs the others (which would skew term-frequency scoring)."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = [(os.path.basename(path), [])]
    for line in lines:
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            sections.append((m.group(1).strip(), []))
        else:
            sections[-1][1].append(line)

    docs: list[Document] = []
    for title, body in sections:
        body_text = "\n".join(body).strip()
        if not body_text:
            continue
        pieces = [body_text]
        if len(body_text) > max_chars:                # split oversized sections
            pieces, cur = [], ""
            for para in re.split(r"\n\s*\n", body_text):
                if len(cur) + len(para) > max_chars and cur:
                    pieces.append(cur.strip())
                    cur = ""
                cur += para + "\n\n"
            if cur.strip():
                pieces.append(cur.strip())
        for i, piece in enumerate(pieces):
            suffix = f"-{i}" if len(pieces) > 1 else ""
            docs.append(Document(
                doc_id=f"{path}#{_slug(title)}{suffix}",
                source=path, title=title, text=piece))
    return docs


def load_corpus(root: str = ".", globs: list[str] | None = None) -> list[Document]:
    """Walk `root` for prep documents and return chunked Documents."""
    globs = globs or DEFAULT_GLOBS
    seen: set[str] = set()
    docs: list[Document] = []
    for pattern in globs:
        for path in glob.glob(os.path.join(root, pattern), recursive=True):
            rel = os.path.relpath(path, root)
            parts = set(rel.split(os.sep))
            if (rel in seen or parts & set(EXCLUDE_DIRS)
                    or os.path.basename(rel) in EXCLUDE_FILES):
                continue
            seen.add(rel)
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except (OSError, UnicodeError):
                continue
            docs.extend(_chunk_markdown(rel, text))
    return docs
