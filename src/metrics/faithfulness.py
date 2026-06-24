"""
faithfulness.py
===============
Faithfulness:       Fraction of answer sentences supported by context.
Hallucination Score: Fraction of answer sentences NOT in context (lower=better).
"""

from __future__ import annotations

from src.metrics.precision_recall import EvalResult
from src.utils.text_utils import coverage, sentence_split


class FaithfulnessMetric:
    """
    A sentence in the answer is 'supported' if a meaningful portion
    of its tokens appear in the retrieved context.
    """

    def __init__(self, threshold: float = 0.7,
                 support_threshold: float = 0.35):
        self.threshold         = threshold
        self.support_threshold = support_threshold

    def evaluate(self, answer: str,
                 context_chunks: list[str]) -> EvalResult:
        sentences = sentence_split(answer)
        if not sentences:
            return EvalResult("faithfulness", 0.0, False, self.threshold,
                              {"error": "empty answer"})

        context_text = " ".join(context_chunks)
        supported    = [s for s in sentences
                        if coverage(s, context_text) >= self.support_threshold]
        unsupported  = [s for s in sentences if s not in supported]
        score        = len(supported) / len(sentences)

        return EvalResult(
            metric    = "faithfulness",
            score     = round(score, 4),
            passed    = score >= self.threshold,
            threshold = self.threshold,
            details   = {
                "total_sentences"  : len(sentences),
                "supported_count"  : len(supported),
                "unsupported"      : unsupported,
            },
        )


class HallucinationMetric:
    """
    Measures the fraction of answer sentences that are NOT grounded in context.
    Score of 0 = perfect (no hallucination).
    Score of 1 = fully hallucinated.
    Passes when hallucination rate ≤ threshold.
    """

    def __init__(self, threshold: float = 0.3,
                 support_threshold: float = 0.25):
        self.threshold         = threshold
        self.support_threshold = support_threshold

    def evaluate(self, answer: str,
                 context_chunks: list[str]) -> EvalResult:
        sentences    = sentence_split(answer)
        if not sentences:
            return EvalResult("hallucination", 0.0, True, self.threshold,
                              {"note": "empty answer — treating as no hallucination"})

        context_text  = " ".join(context_chunks)
        hallucinated  = [s for s in sentences
                         if coverage(s, context_text) < self.support_threshold]
        score         = len(hallucinated) / len(sentences)

        return EvalResult(
            metric    = "hallucination",
            score     = round(score, 4),
            passed    = score <= self.threshold,
            threshold = self.threshold,
            details   = {
                "total_sentences"     : len(sentences),
                "hallucinated_count"  : len(hallucinated),
                "hallucinated_sentences": hallucinated,
            },
        )
