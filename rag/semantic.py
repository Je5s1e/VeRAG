"""Semantic scoring with optional SentenceTransformer backend.

The projection backend is enhanced to be spec-aware:
- Tokens that appear inside ensures/requires/invariant clauses get 4× weight.
- Verus spec keywords (forall, exists, invariant, etc.) get 2× weight.
- Bigrams of adjacent spec tokens are included as features.
- A structural fingerprint encodes the "shape" of spec clauses.
"""

from __future__ import annotations

import hashlib

from dataclasses import dataclass

from .tokenize import tokenize
from .code_analyzer import _extract_spec_clauses

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

# ── Spec-context extraction for enhanced projection ──────────────────────────

# Verus spec keywords that should get boosted embedding weight
_SPEC_KEYWORDS = frozenset({
    "requires", "ensures", "invariant", "decreases", "assert", "proof",
    "spec", "ghost", "trigger", "forall", "exists", "choose", "recommends",
    "lemma", "nonlinear", "calc", "view",
})


def _extract_spec_tokens(text: str) -> list[str]:
    """Return tokens extracted specifically from spec clause bodies."""
    return tokenize(" ".join(_extract_spec_clauses(text).values()))


def _structural_fingerprint(text: str) -> dict[str, int]:
    """Encode structural properties of spec content as named counts."""
    lower = text.lower()
    return {
        "has_forall": int("forall" in lower),
        "has_exists": int("exists" in lower),
        "has_invariant": int("invariant" in lower),
        "has_ensures": int("ensures" in lower),
        "has_requires": int("requires" in lower),
        "has_decreases": int("decreases" in lower),
        "has_overflow": int("overflow" in lower or "underflow" in lower),
        "has_type_cast": int(" as " in lower),
        "forall_count": min(lower.count("forall"), 5),
        "exists_count": min(lower.count("exists"), 5),
        "invariant_count": min(lower.count("invariant"), 5),
    }


@dataclass
class SemanticConfig:
    backend: str = "auto"  # auto | sentence-transformer | projection
    model_name: str = "all-MiniLM-L6-v2"
    proj_dim: int = 512   # increased from 256 for better capacity


def _unit(vec):
    if np is None:
        return vec
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return vec
    return vec / norm


def _projection_sign(feature: str) -> tuple[int, int]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(digest[:4], "little")
    sign = 1 if (digest[4] & 1) == 0 else -1
    return idx, sign


def _add_feature(vec, feature: str, weight: float, dim: int) -> None:
    idx, sign = _projection_sign(feature)
    vec[idx % dim] += weight * sign


def _projection_embed(text: str, dim: int):
    """Spec-aware random projection embedding."""
    if np is None:
        raise RuntimeError("numpy is required for projection embedding")
    vec = np.zeros(dim, dtype=float)

    # 1. Base token features
    tokens = tokenize(text)
    for tok in tokens:
        w = 2.0 if tok in _SPEC_KEYWORDS else 1.0
        _add_feature(vec, tok, w, dim)

    # 2. Token bigrams (captures local context like "ensures forall", "invariant decreases")
    for i in range(len(tokens) - 1):
        bigram = f"bi:{tokens[i]}_{tokens[i+1]}"
        w = 2.5 if (tokens[i] in _SPEC_KEYWORDS or tokens[i+1] in _SPEC_KEYWORDS) else 0.5
        _add_feature(vec, bigram, w, dim)

    # 3. Spec clause tokens with heavy boost
    spec_tokens = _extract_spec_tokens(text)
    for tok in spec_tokens:
        _add_feature(vec, f"spec:{tok}", 4.0, dim)

    # 4. Spec token bigrams
    for i in range(len(spec_tokens) - 1):
        bigram = f"specbi:{spec_tokens[i]}_{spec_tokens[i+1]}"
        _add_feature(vec, bigram, 3.0, dim)

    # 5. Character trigrams of raw text (captures symbol patterns like "<=", "a[i]")
    raw = "".join(ch for ch in text.lower() if ch.isalnum() or ch in " _<>[]()=!+-*/&|")
    for i in range(max(0, len(raw) - 2)):
        tri = raw[i: i + 3]
        _add_feature(vec, f"tri:{tri}", 0.2, dim)

    # 6. Structural fingerprint features
    fp = _structural_fingerprint(text)
    for fname, fval in fp.items():
        if fval > 0:
            _add_feature(vec, f"struct:{fname}:{fval}", 3.0, dim)

    return _unit(vec)


def _cosine(query_vec, matrix):
    if np is None:
        raise RuntimeError("numpy is required for semantic scoring")
    return matrix @ query_vec


class SemanticScorer:
    def __init__(self, config: SemanticConfig):
        if config.backend not in {"auto", "sentence-transformer", "projection"}:
            raise ValueError(f"Unknown embedding backend: {config.backend}")
        if config.proj_dim <= 0:
            raise ValueError("Projection dimension must be positive")
        self.config = config
        self.degraded_reason = ""
        self._backend = "projection"
        self._st_model = None
        if config.backend in ("auto", "sentence-transformer"):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                # local_files_only avoids long network retries in offline environments.
                self._st_model = SentenceTransformer(config.model_name, local_files_only=True)
                self._backend = "sentence-transformer"
            except Exception as exc:
                self.degraded_reason = f"Local model unavailable: {type(exc).__name__}"
                if config.backend == "sentence-transformer":
                    raise
                self._backend = "projection"

    @property
    def backend_name(self) -> str:
        return self._backend

    def embed(self, texts: list[str]):
        if np is None:
            raise RuntimeError("numpy is required for semantic scoring")
        if self._backend == "sentence-transformer":
            emb = self._st_model.encode(texts, normalize_embeddings=True)  # type: ignore
            return np.asarray(emb, dtype=np.float32)
        emb = np.stack([_projection_embed(t, self.config.proj_dim) for t in texts], axis=0)
        return emb

    def score(self, query_text: str, candidate_texts: list[str]) -> list[float]:
        if not candidate_texts:
            return []
        if np is None:
            return [0.0 for _ in candidate_texts]
        q = self.embed([query_text])[0]
        m = self.embed(candidate_texts)
        sims = _cosine(q, m)
        return [float(x) for x in sims.tolist()]
