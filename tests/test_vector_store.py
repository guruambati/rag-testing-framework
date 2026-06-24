"""test_vector_store.py — InMemoryVectorStore tests."""

import pytest
from src.pipeline.ingester import DocumentIngester
from src.pipeline.chunker import TextChunker
from src.pipeline.embedder import EmbeddingEngine
from src.pipeline.vector_store import InMemoryVectorStore
from src.pipeline.chunker import Chunk


def _build_chunks(text: str) -> list[Chunk]:
    ing     = DocumentIngester()
    doc     = ing.ingest_text(text)
    chunker = TextChunker(chunk_size=100, overlap=10, strategy="fixed")
    chunks  = chunker.chunk(doc)
    return EmbeddingEngine().embed_chunks(chunks)


class TestVectorStore:

    def test_add_and_search_returns_results(self):
        store  = InMemoryVectorStore()
        chunks = _build_chunks("Python is a programming language used for data science.")
        store.add(chunks)
        q_emb   = EmbeddingEngine().embed("Python programming")
        results = store.search(q_emb, top_k=3)
        assert len(results) > 0

    def test_results_ranked_descending(self):
        store  = InMemoryVectorStore()
        chunks = _build_chunks("Python is a programming language. " * 10)
        store.add(chunks)
        q_emb   = EmbeddingEngine().embed("Python")
        results = store.search(q_emb, top_k=5)
        scores  = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        store  = InMemoryVectorStore()
        chunks = _build_chunks("Python is a programming language. " * 20)
        store.add(chunks)
        q_emb   = EmbeddingEngine().embed("Python")
        results = store.search(q_emb, top_k=2)
        assert len(results) <= 2

    def test_empty_store_returns_empty_list(self):
        store   = InMemoryVectorStore()
        q_emb   = EmbeddingEngine().embed("test")
        results = store.search(q_emb, top_k=3)
        assert results == []

    def test_chunk_without_embedding_raises(self):
        store = InMemoryVectorStore()
        ing   = DocumentIngester()
        doc   = ing.ingest_text("test")
        bad_chunk = Chunk(chunk_id="c1", doc_id=doc.doc_id,
                          text="text", start_char=0, end_char=4, embedding=None)
        with pytest.raises(ValueError, match="embedding"):
            store.add([bad_chunk])

    def test_score_in_valid_range(self):
        store  = InMemoryVectorStore()
        chunks = _build_chunks("Cloud computing is scalable and cost-effective.")
        store.add(chunks)
        q_emb   = EmbeddingEngine().embed("cloud")
        results = store.search(q_emb, top_k=3)
        assert all(-1.0 <= r.score <= 1.0 for r in results)

    def test_size_reflects_chunks_added(self):
        store  = InMemoryVectorStore()
        chunks = _build_chunks("Some text here for sizing test. " * 5)
        store.add(chunks)
        assert store.size == len(chunks)
