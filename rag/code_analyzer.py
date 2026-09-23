"""Extract structural spec features from Verus/Rust code for query enhancement.

This module identifies:
- Tokens inside ensures/requires/invariant clauses (high semantic weight)
- Proof patterns: min/max element, sum, search, sort, reverse, count, etc.
- Type signatures and key identifiers
- Quantifier shapes (forall/exists bound variables)
"""

from __future__ import annotations

import re
from collections import Counter

from .tokenize import token_counts, tokenize
from .chunking import mask_non_code

# ── Spec clause extraction ────────────────────────────────────────────────────

# Match types like Vec<i32>, Seq<u64>, &[T], usize, nat, int
_TYPE_RE = re.compile(
    r"\bVec\s*<\s*(\w+)\s*>|\bSeq\s*<\s*(\w+)\s*>|\bi8\b|\bi16\b|\bi32\b|\bi64\b"
    r"|\bu8\b|\bu16\b|\bu32\b|\bu64\b|\bi128\b|\bu128\b|\busize\b|\bnat\b|\bisize\b"
)

# ── Proof pattern signatures ──────────────────────────────────────────────────

PROOF_PATTERNS: dict[str, list[str]] = {
    "min_element": ["min", "minimum", "getmini", "min_val", "min_value", "smallest"],
    "max_element": ["max", "maximum", "getmaxi", "max_val", "max_value", "largest", "find_max"],
    "sum_prefix": ["sum", "total", "cumsum", "prefix_sum", "triangle"],
    "linear_search": ["search", "find", "first_e", "index_of", "position", "first_occurrence"],
    "sort_check": ["sorted", "is_sorted", "sort"],
    "reverse": ["reverse", "reversed"],
    "rotate": ["rotate", "rotation", "offset"],
    "count": ["count", "cnt", "num_occ", "occurrences"],
    "unique": ["unique", "distinct", "seq_is_unique"],
    "append": ["append", "push"],
    "abs": ["abs", "absolute", "abs_val", "abs_value", "absx", "aba"],
    "tangent": ["tangent", "intersection", "common", "overlap"],
    "arithmetic_cast": ["u128", "i128", "as u64", "as i32", "as usize"],
    "carre": ["carre", "square", "multiply"],
}


def _extract_spec_clauses(code: str) -> dict[str, str]:
    """Return concatenated text for each spec clause type."""
    clauses: dict[str, list[str]] = {
        "ensures": [],
        "requires": [],
        "invariant": [],
        "decreases": [],
    }
    masked = mask_non_code(code)
    # Walk balanced clause expressions; a quantifier block is not a function body.
    keywords = re.compile(r"\b(ensures|requires|invariant|decreases)\b")
    position = 0
    while True:
        match = keywords.search(masked, position)
        if match is None:
            break
        start = match.end()
        i = start
        depth = 0
        while i < len(masked):
            if depth == 0 and keywords.match(masked, i):
                break
            ch = masked[i]
            if ch == "{" and depth == 0:
                prefix = masked[start:i].rstrip()
                # Quantifier closures commonly use forall|x: T| { ... }.
                if not prefix.endswith("|"):
                    break
            if ch in "({[":
                depth += 1
            elif ch in ")}]":
                if depth == 0:
                    break
                depth -= 1
            i += 1
        clauses[match.group(1)].append(code[start:i].strip())
        position = max(i, match.end())
    return {k: " ".join(v) for k, v in clauses.items()}


def _detect_proof_patterns(code: str) -> list[str]:
    """Return list of detected proof pattern names."""
    code_lower = code.lower()
    detected: list[str] = []
    for name, signals in PROOF_PATTERNS.items():
        for sig in signals:
            if re.search(r"(?<![a-z0-9_])" + re.escape(sig.lower()) + r"(?![a-z0-9_])", code_lower):
                detected.append(name)
                break
    return detected


def _extract_key_identifiers(code: str) -> list[str]:
    """Extract non-trivial function and variable identifiers."""
    # Collect all identifiers ≥ 4 chars that aren't Rust/Verus keywords
    _RUST_KWS = {
        "self", "super", "crate", "true", "false", "return", "while",
        "loop", "break", "continue", "match", "type", "struct", "enum",
        "trait", "where", "move", "async", "await", "dyn", "impl",
        "verus", "vstd", "prelude", "proof", "spec", "exec", "ghost",
        "requires", "ensures", "invariant", "decreases", "assert",
        "forall", "exists", "trigger", "choose", "recommends",
        "with", "from", "into", "some", "none",
    }
    found = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]{3,})\b", code)
    counts: Counter[str] = Counter()
    for tok in found:
        low = tok.lower()
        if low not in _RUST_KWS:
            counts[low] += 1
    # Return identifiers that appear >= 2 times (likely meaningful)
    return [t for t, c in counts.most_common(30) if c >= 2]


def _extract_types(code: str) -> list[str]:
    """Extract primitive and generic types used in the code."""
    types: list[str] = []
    for m in _TYPE_RE.finditer(code):
        types.append(m.group(0).replace(" ", "").lower())
    return list(set(types))


def extract_spec_query_terms(code: str, spec_weight: float = 3.0) -> Counter:
    """
    Extract a weighted query term Counter from Verus code.

    Spec-clause tokens are weighted ``spec_weight`` times normal code tokens.
    Proof pattern names are added as bonus terms.
    """
    weighted: Counter = Counter()

    # 1. Base code tokens (weight = 1.0)
    for tok, cnt in token_counts(code).items():
        weighted[tok] += cnt

    # 2. Spec clause tokens (high weight)
    clauses = _extract_spec_clauses(code)
    spec_text = " ".join(clauses.values())
    for tok, cnt in token_counts(spec_text).items():
        # Additional spec_weight bonus on top of base (already counted in code)
        weighted[tok] += cnt * (spec_weight - 1.0)

    # 3. Proof patterns as extra terms
    patterns = _detect_proof_patterns(code)
    for pat in patterns:
        weighted[pat.replace("_", "")] += 3.0
        for word in pat.split("_"):
            weighted[word] += 2.5

    # 4. Key identifiers (meaningful names)
    for ident in _extract_key_identifiers(code):
        weighted[ident] += 1.5

    # 5. Types used (lower weight, helps match related implementations)
    for t in _extract_types(code):
        weighted[t] += 1.0

    return weighted


def get_proof_pattern_labels(code: str) -> list[str]:
    """Return human-readable proof pattern names detected in code."""
    return _detect_proof_patterns(code)


def get_spec_summary(code: str) -> str:
    """Return a short text summary of spec clauses for semantic embedding."""
    clauses = _extract_spec_clauses(code)
    parts: list[str] = []
    for kw in ("ensures", "requires", "invariant"):
        txt = clauses.get(kw, "").strip()
        if txt:
            parts.append(f"{kw}: {txt[:200]}")
    patterns = _detect_proof_patterns(code)
    if patterns:
        parts.append("proof_patterns: " + " ".join(patterns))
    return "\n".join(parts)
