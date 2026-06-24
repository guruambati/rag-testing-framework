"""
rag_system.py
=============
Orchestrates the full RAG pipeline:
  ingest → chunk → embed → store → retrieve
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline.ingester import DocumentIngester, Document
from src.pipeline.chunker import TextChunker, Chunk
from src.pipeline.embedder import EmbeddingEngine
from src.pipeline.vector_store import InMemoryVectorStore, RetrievalResult


class RAGSystem:
    """
    End-to-end RAG pipeline for testing.

    Usage:
        rag = RAGSystem(chunk_size=400, strategy="sentence")
        rag.add_file("docs/overview.txt")
        rag.add_text("Python is a programming language...")

        results = rag.retrieve("What is Python?", top_k=3)
        context = rag.get_context("What is Python?", top_k=3)
    """

    def __init__(self,
                 chunk_size: int = 400,
                 overlap: int = 40,
                 strategy: str = "sentence",
                 use_real_embeddings: bool = False):
        self._ingester    = DocumentIngester()
        self._chunker     = TextChunker(chunk_size, overlap, strategy)
        self._embedder    = EmbeddingEngine(use_real_model=use_real_embeddings)
        self._store       = InMemoryVectorStore()
        self._doc_chunks: dict[str, list[Chunk]] = {}

    # ── Indexing ──────────────────────────────────────────────

    def add_text(self, text: str, source: str = "inline",
                 metadata: dict | None = None) -> Document:
        doc    = self._ingester.ingest_text(text, source, metadata)
        self._index(doc)
        return doc

    def add_file(self, path: str | Path) -> Document:
        doc = self._ingester.ingest_file(path)
        self._index(doc)
        return doc

    def add_directory(self, directory: str | Path,
                       extensions: list[str] | None = None) -> list[Document]:
        docs = self._ingester.ingest_directory(directory, extensions)
        for doc in docs:
            self._index(doc)
        return docs

    def _index(self, doc: Document) -> None:
        chunks = self._chunker.chunk(doc)
        chunks = self._embedder.embed_chunks(chunks)
        self._store.add(chunks)
        self._doc_chunks[doc.doc_id] = chunks

    # ── Retrieval ─────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        q_emb = self._embedder.embed(query)
        return self._store.search(q_emb, top_k=top_k)

    def get_context(self, query: str, top_k: int = 3) -> str:
        """Returns retrieved chunk texts joined with separator."""
        results = self.retrieve(query, top_k=top_k)
        return "\n\n---\n\n".join(r.chunk.text for r in results)

    # ── Stats ─────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return self._store.size

    @property
    def total_documents(self) -> int:
        return self._ingester.count

    def chunks_for_doc(self, doc_id: str) -> list[Chunk]:
        return self._doc_chunks.get(doc_id, [])

    def is_empty(self) -> bool:
        return self._store.is_empty
