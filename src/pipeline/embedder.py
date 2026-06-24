"""
embedder.py
===========
Produces vector embeddings for text chunks.

Default: deterministic 64-dim hash embedding.
  - No API key required
  - Same input always produces same output
  - Useful for testing retrieval logic without real semantic similarity

Optional: real sentence-transformers (all-MiniLM-L6-v2)
  - Set use_real_model=True
  - Requires: pip install sentence-transformers
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.chunker import Chunk


class EmbeddingEngine:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 use_real_model: bool = False,
                 dim: int = 64):
        self.model_name     = model_name
        self.use_real_model = use_real_model
        self.dim            = dim
        self._model         = None

        if use_real_model:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                self.dim    = self._model.get_sentence_embedding_dimension()
            except ImportError:
                print("[EmbeddingEngine] sentence-transformers not installed."
                      " Falling back to hash embeddings.")
                self.use_real_model = False

    def embed(self, text: str) -> list[float]:
        if self.use_real_model and self._model:
            return self._model.encode(text).tolist()
        return self._hash_embed(text, self.dim)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def embed_chunks(self, chunks: list["Chunk"]) -> list["Chunk"]:
        for chunk in chunks:
            chunk.embedding = self.embed(chunk.text)
        return chunks

    @staticmethod
    def _hash_embed(text: str, dim: int = 64) -> list[float]:
        """
        Deterministic pseudo-embedding using iterated MD5.
        Values are in [-1, 1]. Same text → same vector every time.
        """
        seed = text.encode("utf-8")
        vec: list[float] = []
        while len(vec) < dim:
            seed = hashlib.md5(seed).digest()
            for byte in seed:
                vec.append((byte / 127.5) - 1.0)
        return vec[:dim]
