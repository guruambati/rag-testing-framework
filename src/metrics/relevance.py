"""
relevance.py
============
Answer Relevance: does the answer address the question?
"""

from __future__ import annotations

from src.metrics.precision_recall import EvalResult
from src.utils.text_utils import blend_similarity, jaccard


class AnswerRelevanceMetric:
    """
    Measures how well the answer addresses the original question
    using token overlap similarity.
    """

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold

    def evaluate(self, answer: str, question: str) -> EvalResult:
        score = blend_similarity(answer, question)
        return EvalResult(
            metric    = "answer_relevance",
            score     = score,
            passed    = score >= self.threshold,
            threshold = self.threshold,
            details   = {
                "question_preview": question[:80],
                "answer_preview"  : answer[:80],
            },
        )
