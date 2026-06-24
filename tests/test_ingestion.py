"""test_ingestion.py — DocumentIngester tests."""

import pytest
from src.pipeline.ingester import DocumentIngester, Document


@pytest.fixture
def ingester():
    return DocumentIngester()


class TestIngestion:

    def test_ingest_text_returns_document(self, ingester):
        doc = ingester.ingest_text("Hello world", source="test")
        assert isinstance(doc, Document)
        assert doc.content == "Hello world"

    def test_doc_id_is_deterministic(self, ingester):
        d1 = ingester.ingest_text("same content")
        d2 = ingester.ingest_text("same content")
        assert d1.doc_id == d2.doc_id

    def test_different_content_gives_different_id(self, ingester):
        d1 = ingester.ingest_text("content A")
        d2 = ingester.ingest_text("content B")
        assert d1.doc_id != d2.doc_id

    def test_metadata_preserved(self, ingester):
        doc = ingester.ingest_text("text", metadata={"author": "guru"})
        assert doc.metadata["author"] == "guru"

    def test_multiple_docs_tracked(self, ingester):
        ingester.ingest_text("doc 1")
        ingester.ingest_text("doc 2")
        ingester.ingest_text("doc 3")
        assert ingester.count == 3

    def test_source_stored_on_document(self, ingester):
        doc = ingester.ingest_text("text", source="wiki.txt")
        assert doc.source == "wiki.txt"

    def test_ingest_file(self, ingester, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("File content here")
        doc = ingester.ingest_file(f)
        assert doc.content == "File content here"
        assert doc.metadata["filename"] == "test.txt"

    def test_ingest_directory_filters_by_extension(self, ingester, tmp_path):
        (tmp_path / "a.txt").write_text("Document A")
        (tmp_path / "b.txt").write_text("Document B")
        (tmp_path / "c.json").write_text("{}")
        docs = ingester.ingest_directory(tmp_path, extensions=[".txt"])
        assert len(docs) == 2

    def test_whitespace_stripped_from_content(self, ingester):
        doc = ingester.ingest_text("  hello  ")
        assert doc.content == "hello"
