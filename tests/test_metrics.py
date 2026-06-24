"""test_metrics.py — All RAG evaluation metric tests."""

import pytest
from src.metrics.precision_recall import ContextPrecisionMetric, ContextRecallMetric
from src.metrics.faithfulness import FaithfulnessMetric, HallucinationMetric
from src.metrics.relevance import AnswerRelevanceMetric
from src.metrics.citations import CitationValidationMetric


# ── Context Precision ──────────────────────────────────────────

class TestContextPrecision:

    def test_relevant_chunks_pass(self):
        chunks = [
            "Python is a programming language created by Guido van Rossum.",
            "Python supports object-oriented and functional programming.",
        ]
        result = ContextPrecisionMetric(threshold=0.3).evaluate(
            chunks, "Python is a programming language."
        )
        assert result.passed
        assert result.score > 0
        assert result.metric == "context_precision"

    def test_irrelevant_chunks_fail_high_threshold(self):
        chunks = ["The weather in Tokyo is sunny.", "Cats are mammals."]
        result = ContextPrecisionMetric(threshold=0.9).evaluate(
            chunks, "Python is a programming language."
        )
        assert not result.passed

    def test_empty_chunks_returns_zero(self):
        result = ContextPrecisionMetric().evaluate([], "any answer")
        assert result.score == 0.0
        assert not result.passed

    def test_details_contain_counts(self):
        chunks = ["Python is a language.", "Cloud computing is scalable."]
        result = ContextPrecisionMetric().evaluate(chunks, "Python programming")
        assert "total_retrieved" in result.details
        assert result.details["total_retrieved"] == 2


# ── Context Recall ─────────────────────────────────────────────

class TestContextRecall:

    def test_high_recall_when_gt_chunks_found(self):
        gt = ["Python was created in 1991."]
        retrieved = ["Python was created in 1991 by Guido van Rossum.", "Other info."]
        result = ContextRecallMetric(threshold=0.5).evaluate(retrieved, gt)
        assert result.passed

    def test_zero_recall_when_nothing_matches(self):
        gt = ["Very specific technical detail about Python internals."]
        retrieved = ["Cloud is scalable.", "AI uses neural nets."]
        result = ContextRecallMetric(threshold=0.9).evaluate(retrieved, gt)
        assert not result.passed

    def test_empty_ground_truth_returns_perfect_score(self):
        result = ContextRecallMetric().evaluate(["any chunk"], [])
        assert result.score == 1.0
        assert result.passed

    def test_details_include_missed(self):
        gt = ["Python history fact.", "Python syntax detail."]
        retrieved = ["Cloud info here."]
        result = ContextRecallMetric(threshold=0.9).evaluate(retrieved, gt)
        assert "missed_previews" in result.details


# ── Faithfulness ───────────────────────────────────────────────

class TestFaithfulness:

    def test_grounded_answer_passes(self):
        context = [
            "Python is a high-level programming language. "
            "It was created by Guido van Rossum in 1991."
        ]
        answer = "Python is a programming language created by Guido."
        result = FaithfulnessMetric(threshold=0.4).evaluate(answer, context)
        assert result.score > 0

    def test_hallucinated_answer_low_score(self):
        context = ["The sky is blue on a clear day."]
        answer  = "Quantum entanglement solves polynomial time problems instantly."
        result  = FaithfulnessMetric(threshold=0.9).evaluate(answer, context)
        assert not result.passed

    def test_empty_answer_fails(self):
        result = FaithfulnessMetric().evaluate("", ["context here"])
        assert result.score == 0.0
        assert not result.passed

    def test_details_include_unsupported(self):
        context = ["Python is great."]
        answer  = "Python is great. Also quantum computers are amazing."
        result  = FaithfulnessMetric().evaluate(answer, context)
        assert "unsupported" in result.details

    def test_metric_name_is_faithfulness(self):
        result = FaithfulnessMetric().evaluate("answer", ["context"])
        assert result.metric == "faithfulness"


# ── Hallucination ──────────────────────────────────────────────

class TestHallucination:

    def test_grounded_answer_low_hallucination(self):
        context = ["Python is a programming language that supports OOP."]
        answer  = "Python is a programming language that supports OOP."
        result  = HallucinationMetric(threshold=0.4).evaluate(answer, context)
        assert result.passed

    def test_hallucinated_answer_high_score(self):
        context = ["Dogs are pets."]
        answer  = "Quantum entanglement enables faster-than-light communication between stars."
        result  = HallucinationMetric(threshold=0.1).evaluate(answer, context)
        assert not result.passed

    def test_score_is_between_0_and_1(self):
        context = ["Some context text here about programming."]
        answer  = "This is an answer. Another sentence added here."
        result  = HallucinationMetric().evaluate(answer, context)
        assert 0.0 <= result.score <= 1.0

    def test_empty_answer_passes_with_zero_score(self):
        result = HallucinationMetric().evaluate("", ["context"])
        assert result.score == 0.0
        assert result.passed

    def test_hallucinated_sentences_listed_in_details(self):
        context = ["The sky is blue."]
        answer  = "Quantum teleportation works at room temperature."
        result  = HallucinationMetric().evaluate(answer, context)
        assert "hallucinated_sentences" in result.details


# ── Answer Relevance ───────────────────────────────────────────

class TestAnswerRelevance:

    def test_relevant_answer_passes(self):
        question = "What is Python used for?"
        answer   = "Python is used for web development, data science, and automation."
        result   = AnswerRelevanceMetric(threshold=0.2).evaluate(answer, question)
        assert result.passed

    def test_off_topic_answer_fails(self):
        question = "What is Python used for?"
        answer   = "The capital of France is Paris, located along the Seine river."
        result   = AnswerRelevanceMetric(threshold=0.4).evaluate(answer, question)
        assert not result.passed

    def test_score_between_0_and_1(self):
        result = AnswerRelevanceMetric().evaluate("some answer", "some question")
        assert 0.0 <= result.score <= 1.0

    def test_metric_name_is_answer_relevance(self):
        result = AnswerRelevanceMetric().evaluate("answer", "question")
        assert result.metric == "answer_relevance"


# ── Citation Validation ────────────────────────────────────────

class TestCitationValidation:

    def test_valid_citation_passes(self):
        answer   = "Python is a programming language."
        sources  = ["python_wiki.txt"]
        contents = {"python_wiki.txt": "Python is a high-level programming language used widely."}
        result   = CitationValidationMetric(threshold=0.1).evaluate(
            answer, sources, contents
        )
        assert result.passed

    def test_invalid_citation_fails(self):
        answer   = "The capital of France is Paris."
        sources  = ["python_wiki.txt"]
        contents = {"python_wiki.txt": "Python was created by Guido van Rossum."}
        result   = CitationValidationMetric(threshold=0.9).evaluate(
            answer, sources, contents
        )
        assert not result.passed

    def test_no_citations_returns_perfect_score(self):
        result = CitationValidationMetric().evaluate("answer", [], {})
        assert result.score == 1.0
        assert result.passed

    def test_missing_source_content_fails(self):
        answer   = "Some important claim."
        sources  = ["missing_source.txt"]
        contents = {}
        result   = CitationValidationMetric(threshold=0.9).evaluate(
            answer, sources, contents
        )
        assert not result.passed

    def test_valid_and_invalid_citations_split_correctly(self):
        answer   = "Python is a programming language."
        sources  = ["python.txt", "unrelated.txt"]
        contents = {
            "python.txt":    "Python is a high-level programming language.",
            "unrelated.txt": "The Roman Empire fell in 476 AD.",
        }
        result = CitationValidationMetric(threshold=0.4).evaluate(
            answer, sources, contents
        )
        assert "python.txt" in result.details["valid_citations"]
