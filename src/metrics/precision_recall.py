"""
precision_recall.py
===================
Context Precision and Context Recall metrics for RAG evaluation.

Context Precision:  Of the chunks retrieved, how many were relevant?
Context Recall:     Of all relevant chunks, how many were retrieved?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from src.utils.text_utils import jaccard, coverage


@dataclass
class EvalResult:
    metric: str
    score: float
    passed: bool
    threshold: float
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"EvalResult({self.metric}, score={self.score:.3f}, [{status}])"

    def __bool__(self) -> bool:
        return self.passed


class ContextPrecisionMetric:
    """
    Fraction of retrieved chunks that are relevant to the expected answer.
    A chunk is considered relevant if its token overlap with the
    ground-truth answer exceeds the relevance_threshold.
    """

    def __init__(self, threshold: float = 0.5,
                 relevance_threshold: float = 0.08):
        self.threshold           = threshold
        self.relevance_threshold = relevance_threshold

    def evaluate(self, retrieved_chunks: list[str],
                 ground_truth: str) -> EvalResult:
        if not retrieved_chunks:
            return EvalResult("context_precision", 0.0, False, self.threshold,
                              {"error": "no chunks provided"})

        relevant = [
            c for c in retrieved_chunks
            if jaccard(c, ground_truth) >= self.relevance_threshold
        ]
        score = len(relevant) / len(retrieved_chunks)
        return EvalResult(
            metric    = "context_precision",
            score     = round(score, 4),
            passed    = score >= self.threshold,
            threshold = self.threshold,
            details   = {
                "total_retrieved": len(retrieved_chunks),
                "relevant_count" : len(relevant),
                "irrelevant_previews": [c[:60] for c in retrieved_chunks
                                        if c not in relevant],
            },
        )


class ContextRecallMetric:
    """
    Fraction of ground-truth relevant chunks that were retrieved.
    """

    def __init__(self, threshold: float = 0.7,
                 coverage_threshold: float = 0.4):
        self.threshold          = threshold
        self.coverage_threshold = coverage_threshold

    def evaluate(self, retrieved_chunks: list[str],
                 ground_truth_chunks: list[str]) -> EvalResult:
        if not ground_truth_chunks:
            return EvalResult("context_recall", 1.0, True, self.threshold,
                              {"note": "no ground-truth chunks provided"})

        retrieved_text = " ".join(retrieved_chunks).lower()
        found   = [c for c in ground_truth_chunks
                   if coverage(c, retrieved_text) >= self.coverage_threshold]
        missed  = [c for c in ground_truth_chunks if c not in found]
        score   = len(found) / len(ground_truth_chunks)

        return EvalResult(
            metric    = "context_recall",
            score     = round(score, 4),
            passed    = score >= self.threshold,
            threshold = self.threshold,
            details   = {
                "ground_truth_count": len(ground_truth_chunks),
                "found_count"       : len(found),
                "missed_previews"   : [c[:60] for c in missed],
            },
        )
