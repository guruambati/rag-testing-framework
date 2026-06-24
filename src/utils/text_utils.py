"""
text_utils.py
=============
Shared text processing helpers used across pipeline and metrics.
All pure functions — no side effects.
"""

from __future__ import annotations

import re


def tokenize(text: str) -> set[str]:
    """Lowercase word tokens, stripping punctuation."""
    return set(re.findall(r'\b[a-z0-9]+\b', text.lower()))


def sentence_split(text: str) -> list[str]:
    """Split text into sentences on .!? boundaries."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity between token sets of two strings."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def token_f1(prediction: str, reference: str) -> float:
    """
    Token-level F1 between prediction and reference.
    Same calculation used in SQuAD evaluation.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens  = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common    = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def coverage(needle: str, haystack: str) -> float:
    """
    Fraction of needle tokens found in haystack.
    Used for checking if a sentence is 'grounded' in context.
    """
    tn = tokenize(needle)
    th = tokenize(haystack)
    if not tn:
        return 1.0
    return len(tn & th) / len(tn)


def blend_similarity(a: str, b: str) -> float:
    """Blend of Jaccard and token F1 for a balanced similarity score."""
    return round(0.5 * jaccard(a, b) + 0.5 * token_f1(a, b), 4)
