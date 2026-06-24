"""test_embedder.py — EmbeddingEngine tests."""

import pytest
from src.pipeline.embedder import EmbeddingEngine
from src.pipeline.ingester import DocumentIngester
from src.pipeline.chunker import TextChunker


@pytest.fixture
def embedder():
    return EmbeddingEngine(use_real_model=False, dim=64)


class TestEmbedder:

    def test_embed_returns_list(self, embedder):
        result = embedder.embed("hello world")
        assert isinstance(result, list)
        assert len(result) == 64

    def test_embed_is_deterministic(self, embedder):
        assert embedder.embed("test") == embedder.embed("test")

    def test_different_texts_give_different_embeddings(self, embedder):
        e1 = embedder.embed("Python programming")
        e2 = embedder.embed("Kubernetes cloud")
        assert e1 != e2

    def test_values_in_range(self, embedder):
        emb = embedder.embed("range check")
        assert all(-1.0 <= v <= 1.0 for v in emb)

    def test_embed_batch_correct_count(self, embedder):
        texts = ["one", "two", "three"]
        embs  = embedder.embed_batch(texts)
        assert len(embs) == 3

    def test_embed_chunks_attaches_embeddings(self, embedder):
        ing     = DocumentIngester()
        doc     = ing.ingest_text("Python is great. It is easy to learn.")
        chunker = TextChunker(chunk_size=50, overlap=0, strategy="fixed")
        chunks  = chunker.chunk(doc)
        chunks  = embedder.embed_chunks(chunks)
        assert all(c.embedding is not None for c in chunks)
        assert all(len(c.embedding) == 64 for c in chunks)
