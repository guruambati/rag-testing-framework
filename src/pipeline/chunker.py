"""
chunker.py
==========
Splits Documents into Chunks using three strategies:
  - fixed      : overlapping character windows
  - sentence   : split on .!? boundaries, fill to chunk_size
  - paragraph  : split on blank lines
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.ingester import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)


class TextChunker:

    STRATEGIES = ("fixed", "sentence", "paragraph")

    def __init__(self, chunk_size: int = 500,
                 overlap: int = 50,
                 strategy: str = "sentence"):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        self.chunk_size = chunk_size
        self.overlap    = overlap
        self.strategy   = strategy

    def chunk(self, doc: "Document") -> list[Chunk]:
        if self.strategy == "fixed":
            return self._fixed(doc)
        if self.strategy == "sentence":
            return self._sentence(doc)
        return self._paragraph(doc)

    # ── Fixed ─────────────────────────────────────────────────

    def _fixed(self, doc: "Document") -> list[Chunk]:
        text, chunks, start, idx = doc.content, [], 0, 0
        while start < len(text):
            end  = min(start + self.chunk_size, len(text))
            chunks.append(Chunk(
                chunk_id  = f"{doc.doc_id}_f{idx}",
                doc_id    = doc.doc_id,
                text      = text[start:end],
                start_char= start,
                end_char  = end,
            ))
            start += self.chunk_size - self.overlap
            idx   += 1
        return chunks

    # ── Sentence ──────────────────────────────────────────────

    def _sentence(self, doc: "Document") -> list[Chunk]:
        sentences = re.split(r'(?<=[.!?])\s+', doc.content)
        chunks, buf, start, idx = [], "", 0, 0

        for sent in sentences:
            if len(buf) + len(sent) > self.chunk_size and buf:
                chunks.append(Chunk(
                    chunk_id  = f"{doc.doc_id}_s{idx}",
                    doc_id    = doc.doc_id,
                    text      = buf.strip(),
                    start_char= start,
                    end_char  = start + len(buf),
                ))
                start += max(0, len(buf) - self.overlap)
                buf    = buf[-self.overlap:] + " " + sent if self.overlap else sent
                idx   += 1
            else:
                buf += (" " if buf else "") + sent

        if buf.strip():
            chunks.append(Chunk(
                chunk_id  = f"{doc.doc_id}_s{idx}",
                doc_id    = doc.doc_id,
                text      = buf.strip(),
                start_char= start,
                end_char  = start + len(buf),
            ))
        return chunks

    # ── Paragraph ─────────────────────────────────────────────

    def _paragraph(self, doc: "Document") -> list[Chunk]:
        paragraphs = re.split(r'\n\s*\n', doc.content)
        chunks, offset = [], 0
        for idx, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                chunks.append(Chunk(
                    chunk_id  = f"{doc.doc_id}_p{idx}",
                    doc_id    = doc.doc_id,
                    text      = para,
                    start_char= offset,
                    end_char  = offset + len(para),
                ))
            offset += len(para) + 2
        return chunks
