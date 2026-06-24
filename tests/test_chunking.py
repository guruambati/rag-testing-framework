"""test_chunking.py — TextChunker tests across all three strategies."""

import pytest
from src.pipeline.ingester import DocumentIngester
from src.pipeline.chunker import TextChunker


@pytest.fixture
def ingester():
    return DocumentIngester()

LONG_TEXT = "Python is a programming language. " * 50


class TestFixedChunking:

    def test_fixed_produces_multiple_chunks(self, ingester):
        doc     = ingester.ingest_text(LONG_TEXT)
        chunker = TextChunker(chunk_size=100, overlap=10, strategy="fixed")
        chunks  = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_size(self, ingester):
        doc     = ingester.ingest_text(LONG_TEXT)
        limit   = 120
        chunker = TextChunker(chunk_size=limit, overlap=0, strategy="fixed")
        chunks  = chunker.chunk(doc)
        assert all(len(c.text) <= limit for c in chunks)

    def test_chunk_ids_are_unique(self, ingester):
        doc     = ingester.ingest_text(LONG_TEXT)
        chunker = TextChunker(chunk_size=100, overlap=10, strategy="fixed")
        chunks  = chunker.chunk(doc)
        ids     = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_all_chunks_reference_parent_doc(self, ingester):
        doc     = ingester.ingest_text(LONG_TEXT)
        chunker = TextChunker(strategy="fixed")
        chunks  = chunker.chunk(doc)
        assert all(c.doc_id == doc.doc_id for c in chunks)

    def test_short_doc_produces_single_chunk(self, ingester):
        doc     = ingester.ingest_text("Short text.")
        chunker = TextChunker(chunk_size=500, overlap=50, strategy="fixed")
        chunks  = chunker.chunk(doc)
        assert len(chunks) == 1


class TestSentenceChunking:

    def test_sentence_strategy_produces_chunks(self, ingester):
        text    = "First sentence. Second sentence. Third sentence. " * 10
        doc     = ingester.ingest_text(text)
        chunker = TextChunker(chunk_size=80, overlap=0, strategy="sentence")
        chunks  = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_chunk_text_is_non_empty(self, ingester):
        doc     = ingester.ingest_text(LONG_TEXT)
        chunker = TextChunker(strategy="sentence")
        chunks  = chunker.chunk(doc)
        assert all(c.text.strip() for c in chunks)


class TestParagraphChunking:

    def test_paragraph_splits_on_blank_lines(self, ingester):
        text    = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        doc     = ingester.ingest_text(text)
        chunker = TextChunker(strategy="paragraph")
        chunks  = chunker.chunk(doc)
        assert len(chunks) == 3

    def test_paragraph_content_preserved(self, ingester):
        text    = "First block.\n\nSecond block."
        doc     = ingester.ingest_text(text)
        chunker = TextChunker(strategy="paragraph")
        chunks  = chunker.chunk(doc)
        assert "First block" in chunks[0].text
        assert "Second block" in chunks[1].text


class TestChunkerValidation:

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            TextChunker(strategy="unknown")

    def test_start_end_chars_are_valid(self, ingester):
        doc     = ingester.ingest_text(LONG_TEXT)
        chunker = TextChunker(chunk_size=100, overlap=10, strategy="fixed")
        chunks  = chunker.chunk(doc)
        for c in chunks:
            assert c.start_char >= 0
            assert c.end_char > c.start_char
