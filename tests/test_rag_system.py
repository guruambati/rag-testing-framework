"""test_rag_system.py — End-to-end RAGSystem tests."""

import pytest
from src.pipeline.rag_system import RAGSystem


class TestRAGSystem:

    def test_add_text_increases_chunk_count(self, empty_rag):
        before = empty_rag.total_chunks
        empty_rag.add_text("Python is a programming language. " * 20)
        assert empty_rag.total_chunks > before

    def test_retrieve_returns_results(self, loaded_rag):
        results = loaded_rag.retrieve("What is Python?", top_k=3)
        assert len(results) > 0

    def test_retrieve_top_k_respected(self, loaded_rag):
        results = loaded_rag.retrieve("programming language", top_k=2)
        assert len(results) <= 2

    def test_python_query_returns_python_chunks(self, loaded_rag):
        results = loaded_rag.retrieve("Python programming language Guido", top_k=3)
        combined = " ".join(r.chunk.text for r in results).lower()
        assert "python" in combined

    def test_cloud_query_finds_cloud_content(self, loaded_rag):
        results = loaded_rag.retrieve("AWS Kubernetes cloud computing", top_k=3)
        combined = " ".join(r.chunk.text for r in results).lower()
        assert any(kw in combined for kw in ["cloud", "aws", "kubernetes"])

    def test_get_context_returns_string(self, loaded_rag):
        ctx = loaded_rag.get_context("What is machine learning?", top_k=2)
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_total_documents_count(self, loaded_rag):
        assert loaded_rag.total_documents == 3

    def test_chunks_for_doc_correct(self, empty_rag):
        doc = empty_rag.add_text("Python is a great language. " * 10, source="py.txt")
        chunks = empty_rag.chunks_for_doc(doc.doc_id)
        assert len(chunks) > 0
        assert all(c.doc_id == doc.doc_id for c in chunks)

    def test_is_empty_true_before_add(self, empty_rag):
        assert empty_rag.is_empty()

    def test_is_empty_false_after_add(self, empty_rag):
        empty_rag.add_text("some content")
        assert not empty_rag.is_empty()

    def test_add_file(self, empty_rag, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Python is used for web development and automation. " * 10)
        doc = empty_rag.add_file(f)
        assert empty_rag.total_documents == 1
        assert empty_rag.total_chunks > 0
