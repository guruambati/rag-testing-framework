"""
citations.py
============
Citation Validation: do cited sources actually support the answer?
"""

from __future__ import annotations

from src.metrics.precision_recall import EvalResult
from src.utils.text_utils import coverage


class CitationValidationMetric:
    """
    For each cited source, checks whether the answer content
    is grounded in that source's text.
    """

    def __init__(self, threshold: float = 0.6,
                 coverage_threshold: float = 0.12):
        self.threshold          = threshold
        self.coverage_threshold = coverage_threshold

    def evaluate(self, answer: str,
                 cited_sources: list[str],
                 source_contents: dict[str, str]) -> EvalResult:
        if not cited_sources:
            return EvalResult("citation_validation", 1.0, True, self.threshold,
                              {"note": "no citations to validate"})

        valid, invalid = [], []
        for src in cited_sources:
            content = source_contents.get(src, "")
            if coverage(answer[:300], content) >= self.coverage_threshold:
                valid.append(src)
            else:
                invalid.append(src)

        score = len(valid) / len(cited_sources)
        return EvalResult(
            metric    = "citation_validation",
            score     = round(score, 4),
            passed    = score >= self.threshold,
            threshold = self.threshold,
            details   = {
                "valid_citations"  : valid,
                "invalid_citations": invalid,
            },
        )
