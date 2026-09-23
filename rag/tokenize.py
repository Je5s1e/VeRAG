"""Tokenization and lightweight NLP helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "for",
    "as",
    "by",
    "from",
    "with",
    "that",
    "this",
    "it",
    "be",
    "at",
    "if",
    "then",
    "else",
    "let",
    "mut",
    "pub",
    "use",
    "mod",
    "impl",
    "fn",
}


def normalize_token(token: str) -> str:
    value = token.strip().lower()
    return {"invariants": "invariant", "loops": "loop", "preconditions": "precondition",
            "postconditions": "postcondition", "quantifiers": "quantifier"}.get(value, value)


def tokenize(text: str) -> list[str]:
    tokens = [normalize_token(t) for t in TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def token_counts(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def unique_tokens(text: str) -> set[str]:
    return set(tokenize(text))


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    aset = set(a)
    bset = set(b)
    if not aset or not bset:
        return 0.0
    inter = len(aset & bset)
    union = len(aset | bset)
    return inter / max(1, union)
