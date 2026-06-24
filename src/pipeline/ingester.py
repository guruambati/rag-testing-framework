"""
ingester.py
===========
Loads raw text or files into Document objects for the RAG pipeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    content: str
    source: str
    metadata: dict = field(default_factory=dict)


class DocumentIngester:

    def __init__(self):
        self._documents: list[Document] = []

    def ingest_text(self, text: str, source: str = "inline",
                    metadata: dict | None = None) -> Document:
        text   = text.strip()
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        doc    = Document(doc_id=doc_id, content=text,
                          source=source, metadata=metadata or {})
        self._documents.append(doc)
        return doc

    def ingest_file(self, path: str | Path) -> Document:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        return self.ingest_text(
            text, source=str(path), metadata={"filename": path.name}
        )

    def ingest_directory(self, directory: str | Path,
                          extensions: list[str] | None = None) -> list[Document]:
        directory  = Path(directory)
        extensions = extensions or [".txt", ".md"]
        docs = []
        for f in sorted(directory.iterdir()):
            if f.suffix.lower() in extensions:
                docs.append(self.ingest_file(f))
        return docs

    @property
    def documents(self) -> list[Document]:
        return list(self._documents)

    @property
    def count(self) -> int:
        return len(self._documents)
