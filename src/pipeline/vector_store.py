"""
vector_store.py
===============
In-memory vector store with cosine similarity search.
No external dependencies — uses pure Python math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.chunker import Chunk


@dataclass
class RetrievalResult:
    chunk: "Chunk"
    score: float
    rank: int


class InMemoryVectorStore:

    def __init__(self):
        self._chunks: list["Chunk"] = []

    def add(self, chunks: list["Chunk"]) -> None:
        for c in chunks:
            if c.embedding is None:
                raise ValueError(
                    f"Chunk '{c.chunk_id}' has no embedding. "
                    "Run EmbeddingEngine.embed_chunks() first."
                )
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float],
               top_k: int = 5) -> list[RetrievalResult]:
        if not self._chunks:
            return []
        scored = [
            (chunk, self._cosine(query_embedding, chunk.embedding))
            for chunk in self._chunks
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievalResult(chunk=c, score=round(s, 6), rank=i + 1)
            for i, (c, s) in enumerate(scored[:top_k])
        ]

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot    = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
