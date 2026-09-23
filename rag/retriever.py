"""Hybrid retriever for Verus code+error queries."""

from __future__ import annotations

import json
import math
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .code_analyzer import extract_spec_query_terms, get_proof_pattern_labels, get_spec_summary
from .error_cleaner import clean_error_text, detect_error_categories, expand_terms_from_categories
from .semantic import SemanticConfig, SemanticScorer
from .storage import resolve_snapshot
from .tokenize import jaccard_similarity, token_counts, unique_tokens

K1 = 1.5
B = 0.75
RRF_K = 60

VERUS_HINT_KEYWORDS = {
    "requires",
    "ensures",
    "invariant",
    "decreases",
    "assert",
    "recommends",
    "proof",
    "spec",
    "ghost",
    "trigger",
    "forall",
    "exists",
    "nonlinear_arith",
    "choose",
    "loop",
    "while",
    "state",
    "lemma",
    "calc",
    "view",
}

ERROR_PATTERN_HINTS: dict[str, list[str]] = {
    "decreases": ["decreases", "termination", "recursive", "measure"],
    "invariant": ["invariant", "loop", "while", "state"],
    "trigger": ["trigger", "forall", "quantifier"],
    "assertion failure": ["assert", "requires", "ensures"],
    "assertion failed": ["assert", "proof", "lemma", "calc"],
    "postcondition": ["ensures", "postcondition"],
    "precondition": ["requires", "precondition"],
    "cannot prove": ["lemma", "proof", "assert", "invariant"],
    "might not hold": ["invariant", "assert", "lemma"],
    "possibly not satisfied": ["requires", "ensures", "assert"],
    "recommendation not met": ["recommends", "requires", "proof"],
    "trigger may be insufficient": ["trigger", "forall", "exists", "quantifier"],
    "assert_by": ["assert_by", "proof", "lemma"],
    "failed to simplify": ["reveal", "spec", "lemma"],
    "overflow": ["overflow", "underflow", "cast", "as", "arithmetic", "nat", "int"],
    "mismatched types": ["type", "cast", "mode", "exec", "spec", "view", "ghost"],
    "no associated item": ["vstd", "seq", "vec", "prelude", "use", "import"],
    "type annotations": ["type", "infer", "mode", "cast"],
}

SOURCE_PRIOR = {
    "project": 0.16,
    "tutorial": 0.14,
    "pdf": 0.28,
}

QUERY_FIELD_WEIGHTS = {
    "query": 1.0,
    "code": 1.25,
    "error": 1.6,
}

# These path segments indicate noisy / off-topic tutorial documents
TUTORIAL_NOISE_PATH_PARTS = (
    "getting_started",
    "install",
    "index.md",
    "vscode",
    "cmd_line",
    "cmdline",
    "changelog",
    "contributing",
    "faq",
)

PATH_HINTS_BY_ERROR = {
    "invariant": ("invariants", "while", "properties", "state_machines", "quantproofs", "loop"),
    "precondition": ("requires-ensures", "mut-ref", "precondition", "spec", "requires"),
    "postcondition": ("ensures", "while", "proof", "quants", "postcondition"),
    "overflow": ("integers", "nonlinear", "arithmetic", "cast", "overflow", "integer"),
    "type_mismatch": ("integers", "integer", "reference-as", "modes", "exec_spec", "types", "cast", "usize"),
    "assertion": ("assert", "proof", "lemma", "calc", "assert_by"),
    "missing_item": ("vstd", "seq", "vec", "prelude", "stdlib", "vstd_api"),
    "mode_error": ("modes", "exec_spec", "ghost", "tracked"),
    "old_ref": ("mut-ref", "requires-ensures", "reference", "old", "borrow"),
}


@dataclass
class RetrievedChunk:
    score: float
    group_score: float
    id: str
    path: str
    source_type: str
    source_group: str
    lang: str
    line_start: int
    line_end: int
    page: int
    title: str
    tags: list[str]
    source_prior: float
    lexical_score: float
    semantic_score: float
    rrf_score: float
    snippet: str
    text: str
    symbol: str
    chunk_kind: str
    includes: list[dict]


def _load_chunks(index_dir: Path) -> list[dict]:
    chunks_file = index_dir / "chunks.jsonl"
    chunks: list[dict] = []
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _build_bm25_stats(chunks: list[dict]) -> tuple[list[Counter[str]], dict[str, int], float]:
    tf_list: list[Counter[str]] = []
    df: dict[str, int] = defaultdict(int)
    doc_lens: list[int] = []

    for ch in chunks:
        tf = token_counts(ch["text"])
        tf_list.append(tf)
        doc_lens.append(sum(tf.values()))
        for term in tf.keys():
            df[term] += 1

    avg_len = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
    return tf_list, dict(df), avg_len


def _bm25(
    query_terms: Counter[str],
    tf: Counter[str],
    df: dict[str, int],
    n_docs: int,
    avg_len: float,
) -> float:
    dl = sum(tf.values())
    if dl == 0 or avg_len == 0:
        return 0.0
    score = 0.0
    for term, q_weight in query_terms.items():
        f = tf.get(term, 0)
        if f == 0:
            continue
        n_q = df.get(term, 0)
        idf = math.log((n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)
        denom = f + K1 * (1 - B + B * dl / avg_len)
        score += q_weight * idf * (f * (K1 + 1)) / denom
    return score


def _extract_hints(error_text: str) -> set[str]:
    error_l = error_text.lower()
    hints = set()
    for pat, kws in ERROR_PATTERN_HINTS.items():
        if pat in error_l:
            hints.update(kws)
    return hints


def _path_adjustment(path: str, source_group: str, error_categories: list[str]) -> float:
    path_l = path.lower()
    adjust = 0.0
    if source_group == "tutorial":
        if any(part in path_l for part in TUTORIAL_NOISE_PATH_PARTS):
            adjust -= 0.30
    for cat in error_categories:
        for kw in PATH_HINTS_BY_ERROR.get(cat, ()):
            if kw in path_l:
                adjust += 0.10
    return adjust


def _extract_repo_name(path: str) -> str:
    """Extract top-level project/repo name from a path like 'projects/foo/...'."""
    parts = Path(path).parts
    for i, p in enumerate(parts):
        if p in ("projects", "tutorial"):
            if i + 1 < len(parts):
                return parts[i + 1]
    return parts[0] if parts else "unknown"


def _group_boost(
    chunk: dict, query_tokens: set[str], hint_tokens: set[str], error_categories: list[str],
    text_tokens: set[str] | None = None,
) -> float:
    if text_tokens is None:
        text_tokens = unique_tokens(chunk["text"])
    jacc = jaccard_similarity(query_tokens, text_tokens)

    source_group = chunk.get("source_group", "tutorial")
    boost = SOURCE_PRIOR.get(source_group, 0.0)

    verus_overlap = len((query_tokens | hint_tokens) & VERUS_HINT_KEYWORDS & text_tokens)
    boost += min(0.25, verus_overlap * 0.05)
    boost += jacc * 0.6
    boost += min(0.10, len(chunk.get("tags", [])) * 0.01)
    boost += _path_adjustment(chunk.get("path", ""), source_group, error_categories)
    return boost


def _trim_snippet(text: str, max_len: int = 480) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _dedup_key(item: RetrievedChunk) -> tuple[str, int]:
    return item.path, item.page


def _is_near_dup(selected: list[RetrievedChunk], candidate: RetrievedChunk) -> bool:
    c_tokens = set(candidate.text.lower().split())
    for existing in selected:
        if existing.id == candidate.id:
            return True
        if _dedup_key(existing) == _dedup_key(candidate):
            overlap = max(0, min(existing.line_end, candidate.line_end) - max(existing.line_start, candidate.line_start) + 1)
            shorter = min(existing.line_end - existing.line_start + 1, candidate.line_end - candidate.line_start + 1)
            if overlap / max(1, shorter) > 0.8:
                return True
        if jaccard_similarity(c_tokens, existing.text.lower().split()) > 0.95:
            return True
    return False


def _pick_diversified(
    ranked: list[RetrievedChunk],
    top_k: int,
    group_minimum: dict[str, int],
    max_per_repo: int = 2,
) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    grouped_selected: dict[str, int] = defaultdict(int)
    repo_selected: dict[str, int] = defaultdict(int)
    grouped_candidates: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for item in ranked:
        grouped_candidates[item.source_group].append(item)

    def _can_add(cand: RetrievedChunk) -> bool:
        if _is_near_dup(selected, cand):
            return False
        # Per-repo limit applies only to project chunks
        if cand.source_group == "project":
            repo = _extract_repo_name(cand.path)
            if repo_selected[repo] >= max_per_repo:
                return False
        return True

    def _record(cand: RetrievedChunk) -> None:
        selected.append(cand)
        grouped_selected[cand.source_group] += 1
        if cand.source_group == "project":
            repo = _extract_repo_name(cand.path)
            repo_selected[repo] += 1

    # Phase 1: satisfy per-group minimums.
    for group, need in group_minimum.items():
        if need <= 0:
            continue
        for cand in grouped_candidates.get(group, []):
            if len(selected) >= top_k:
                break
            if not _can_add(cand):
                continue
            _record(cand)
            if grouped_selected[group] >= need:
                break

    # Phase 2: fill remaining slots from global rank.
    for cand in ranked:
        if len(selected) >= top_k:
            break
        if not _can_add(cand):
            continue
        _record(cand)
    return selected


def _build_weighted_query_terms(
    query_text: str,
    code_text: str,
    error_text: str,
    proof_patterns: list[str],
) -> Counter[str]:
    weighted: Counter = Counter()
    cleaned_error = clean_error_text(error_text)
    error_categories = detect_error_categories(cleaned_error)
    cat_terms = expand_terms_from_categories(error_categories)

    # Base fields
    blocks = {
        "query": query_text.strip(),
        "error": cleaned_error.strip(),
    }
    for name, text in blocks.items():
        if not text:
            continue
        w = QUERY_FIELD_WEIGHTS[name]
        for term, cnt in token_counts(text).items():
            weighted[term] += cnt * w

    # Code: use spec-aware analysis instead of plain tokenization
    if code_text.strip():
        spec_terms = extract_spec_query_terms(code_text, spec_weight=3.5)
        for term, weight in spec_terms.items():
            weighted[term] += weight * QUERY_FIELD_WEIGHTS["code"]

    # Error hint tokens
    hints = _extract_hints(cleaned_error)
    for term in hints:
        weighted[term] += 2.0

    # Category expansion terms
    for term in cat_terms:
        weighted[term] += 2.2

    # Proof patterns from code analysis
    for pat in proof_patterns:
        weighted[pat.replace("_", "")] += 3.5
        for word in pat.split("_"):
            weighted[word] += 2.5

    return weighted


@lru_cache(maxsize=4)
def _load_cached_bm25(index_dir: str) -> tuple[list[dict], list[Counter[str]], dict[str, int], float]:
    path = Path(index_dir)
    chunks = _load_chunks(path)
    lexical_file = path / "lexical.json"
    if lexical_file.exists():
        tf_list = [Counter(row) for row in json.loads(lexical_file.read_text())]
        if len(tf_list) != len(chunks):
            raise ValueError("Lexical index/chunk count mismatch; rebuild index")
        df = Counter(term for tf in tf_list for term in tf)
        avg_len = sum(sum(tf.values()) for tf in tf_list) / max(1, len(tf_list))
    else:
        tf_list, df, avg_len = _build_bm25_stats(chunks)
    return chunks, tf_list, df, avg_len


@lru_cache(maxsize=4)
def _load_vectors(snapshot: str):
    import numpy as np
    return np.load(Path(snapshot) / "vectors.npy", mmap_mode="r", allow_pickle=False)


# Cache initialization can race even with lru_cache; serialize model construction.
_scorer_lock: threading.Lock = threading.Lock()
_scorer_cache: dict = {}



def _get_semantic_scorer(backend: str, model_name: str, proj_dim: int) -> SemanticScorer:
    key = (backend, model_name, proj_dim)
    # Fast path for an initialized model.
    scorer = _scorer_cache.get(key)
    if scorer is not None:
        return scorer
    # Recheck under the lock before loading the model.
    with _scorer_lock:
        scorer = _scorer_cache.get(key)
        if scorer is None:
            scorer = SemanticScorer(
                SemanticConfig(
                    backend=backend,
                    model_name=model_name,
                    proj_dim=proj_dim,
                )
            )
            _scorer_cache[key] = scorer
    return scorer


def _rrf_fuse(
    lexical_ranked_indices: list[int],
    semantic_ranked_indices: list[int],
    lexical_weight: float,
    semantic_weight: float,
) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for r, idx in enumerate(lexical_ranked_indices if lexical_weight > 0 else [], 1):
        scores[idx] += lexical_weight / (RRF_K + r)
    for r, idx in enumerate(semantic_ranked_indices if semantic_weight > 0 else [], 1):
        scores[idx] += semantic_weight / (RRF_K + r)
    return dict(scores)


def _retrieve_index(
    index_dir: str | Path,
    query_text: str,
    code_text: str = "",
    error_text: str = "",
    top_k: int = 12,
    per_group_k: int = 6,
    min_project: int = 0,
    min_tutorial: int = 0,
    min_pdf: int = 0,
    semantic_backend: str = "auto",
    semantic_model: str | None = None,
    semantic_proj_dim: int | None = None,
    semantic_candidate_k: int = 1200,
    lexical_rrf_weight: float = 1.0,
    semantic_rrf_weight: float | None = None,
    max_context_chars: int = 24000,
) -> dict:
    if top_k <= 0 or per_group_k <= 0 or semantic_candidate_k <= 0:
        raise ValueError("top_k, per_group_k and semantic_candidate_k must be positive")
    if min(min_project, min_tutorial, min_pdf, max_context_chars) < 0:
        raise ValueError("Minimum counts and context budget must be nonnegative")
    if semantic_backend not in {"none", "auto", "projection", "sentence-transformer"}:
        raise ValueError(f"Unknown semantic backend: {semantic_backend}")
    if not all(math.isfinite(w) and w >= 0 for w in (lexical_rrf_weight, semantic_rrf_weight) if w is not None):
        raise ValueError("RRF weights must be finite and nonnegative")
    snapshot, metadata = resolve_snapshot(index_dir)
    # Legacy indexes bypass the cache; immutable generations have distinct paths.
    index_path = str(snapshot)
    if not metadata.get("generation"):
        _load_cached_bm25.cache_clear()

    chunks, tf_list, df, avg_len = _load_cached_bm25(index_path)


    cleaned_error = clean_error_text(error_text)
    error_categories = detect_error_categories(cleaned_error)
    proof_patterns = get_proof_pattern_labels(code_text) if code_text.strip() else []
    query_tf = _build_weighted_query_terms(query_text, code_text, cleaned_error, proof_patterns)
    query_terms = list(query_tf.keys())
    query_tokens = set(query_terms)
    hint_tokens = _extract_hints(cleaned_error)

    lexical_records: list[tuple[int, float, float, float]] = []
    n_docs = len(chunks)
    for idx, (ch, tf) in enumerate(zip(chunks, tf_list)):
        base = _bm25(query_tf, tf, df, n_docs, avg_len)
        if base <= 0:
            continue
        boost = _group_boost(ch, query_tokens, hint_tokens, error_categories, set(tf))
        lexical = base + boost
        lexical_records.append((idx, lexical, base, boost))

    lexical_records.sort(key=lambda x: x[1], reverse=True)
    lexical_ranked_indices = [idx for idx, _, _, _ in lexical_records]

    # Build semantic query incorporating spec summary for code-aware comparison
    spec_summary = get_spec_summary(code_text) if code_text.strip() else ""
    semantic_query = "\n".join(
        filter(None, [query_text.strip(), spec_summary, cleaned_error.strip()])
    ).strip()

    semantic_scores: dict[int, float] = {}
    semantic_ranked_indices: list[int] = []
    semantic_backend_used = "disabled"
    degraded_reason = ""
    embedding = metadata.get("embedding", {})
    if semantic_backend != "none" and semantic_query and chunks:
        if embedding.get("backend", "none") == "none":
            if semantic_backend != "auto":
                raise ValueError("Index has no persisted vectors; rebuild with the requested backend")
            degraded_reason = "Index has no persisted vectors; lexical retrieval only"
        else:
            stored_backend = embedding["backend"]
            if semantic_model is not None and semantic_model != embedding["model"]:
                raise ValueError("Requested model differs from index; rebuild the index")
            if semantic_proj_dim is not None and semantic_proj_dim != embedding["projection_dimension"]:
                raise ValueError("Requested projection dimension differs from index; rebuild the index")
            if semantic_backend != "auto" and semantic_backend != stored_backend:
                raise ValueError("Requested backend differs from index; rebuild or use auto")
            try:
                scorer = _get_semantic_scorer(
                    stored_backend, embedding["model"], embedding["projection_dimension"]
                )
                vectors = _load_vectors(index_path)
                query_vector = scorer.embed([semantic_query])[0]
                if vectors.shape != (len(chunks), len(query_vector)):
                    raise ValueError("Vector index/chunk dimension mismatch; rebuild index")
                similarities = vectors @ query_vector
                # Independent corpus-wide retrieval, including lexical misses.
                semantic_ranked_indices = sorted(
                    (i for i, score in enumerate(similarities) if score > 0),
                    key=lambda i: float(similarities[i]), reverse=True,
                )[:semantic_candidate_k]
                semantic_scores = {i: float(similarities[i]) for i in semantic_ranked_indices}
                semantic_backend_used = scorer.backend_name
                degraded_reason = embedding.get("degraded_reason", "")
            except Exception as exc:
                if semantic_backend != "auto":
                    raise
                semantic_backend_used = "failed"
                degraded_reason = f"{type(exc).__name__}: {exc}"

    # Feature hashing is a weaker signal than a learned embedding model.
    effective_semantic_weight = semantic_rrf_weight
    if effective_semantic_weight is None:
        effective_semantic_weight = 0.25 if semantic_backend_used == "projection" else 0.9
    fused_scores = _rrf_fuse(
        lexical_ranked_indices,
        semantic_ranked_indices,
        lexical_weight=lexical_rrf_weight,
        semantic_weight=effective_semantic_weight if semantic_ranked_indices else 0.0,
    )

    lexical_by_idx = {idx: (lex, base, boost) for idx, lex, base, boost in lexical_records}
    ranked_indices = sorted(
        fused_scores.keys(),
        key=lambda i: (fused_scores.get(i, 0.0), lexical_by_idx.get(i, (0.0, 0.0, 0.0))[0]),
        reverse=True,
    )
    scored: list[RetrievedChunk] = []
    for idx in ranked_indices:
        ch = chunks[idx]
        lexical, _, boost = lexical_by_idx.get(idx, (0.0, 0.0, 0.0))
        sem = semantic_scores.get(idx, 0.0)
        rrf = fused_scores.get(idx, 0.0)
        final = rrf
        scored.append(
            RetrievedChunk(
                score=round(final, 6),
                group_score=round(boost, 6),
                id=ch["id"],
                path=ch["path"],
                source_type=ch["source_type"],
                source_group=ch["source_group"],
                lang=ch["lang"],
                line_start=ch["line_start"],
                line_end=ch["line_end"],
                page=ch.get("page", 0),
                title=ch.get("title", ""),
                tags=ch.get("tags", []),
                source_prior=SOURCE_PRIOR.get(ch["source_group"], 0.0),
                lexical_score=round(lexical, 6),
                semantic_score=round(sem, 6),
                rrf_score=round(rrf, 6),
                snippet=_trim_snippet(ch["text"]),
                text=ch["text"],
                symbol=ch.get("symbol", ""),
                chunk_kind=ch.get("chunk_kind", "text"),
                includes=ch.get("includes", []),
            )
        )

    has_pdf = any(ch.get("source_group") == "pdf" for ch in chunks)
    group_minimum = {
        "project": max(0, min_project),
        "tutorial": max(0, min_tutorial),
        "pdf": max(0, min_pdf if has_pdf else 0),
    }
    top_all = _pick_diversified(scored, top_k=top_k, group_minimum=group_minimum)
    top_all.sort(key=lambda item: item.score, reverse=True)

    grouped: dict[str, list[dict]] = {"project": [], "tutorial": [], "pdf": []}
    for item in scored:
        bucket = grouped.get(item.source_group)
        if bucket is None:
            continue
        if len(bucket) < per_group_k:
            bucket.append(item.__dict__)

    result = {
        "index_generation": metadata.get("generation", "legacy"),
        "results": [x.__dict__ for x in top_all],
        "grouped": grouped,
        "query_terms": query_terms[:60],
        "hint_terms": sorted(hint_tokens),
        "proof_patterns": proof_patterns,
        "error_categories": error_categories,
        "cleaned_error_preview": cleaned_error[:600],
        "group_minimum": group_minimum,
        "retrieval_debug": {
            "semantic_backend": semantic_backend_used,
            "degraded_reason": degraded_reason,
            "semantic_only_candidates": len(set(semantic_ranked_indices) - set(lexical_ranked_indices)),
            "lexical_candidates": len(lexical_ranked_indices),
            "semantic_candidates": len(semantic_ranked_indices),
            "rrf_k": RRF_K,
            "semantic_rrf_weight": effective_semantic_weight if semantic_ranked_indices else 0.0,
            "spec_summary_preview": spec_summary[:200],
        },
    }
    return result


def query_index(
    index_dir: str | Path,
    query_text: str,
    code_text: str = "",
    error_text: str = "",
    top_k: int = 12,
    per_group_k: int = 6,
    min_project: int = 0,
    min_tutorial: int = 0,
    min_pdf: int = 0,
    semantic_backend: str = "auto",
    semantic_model: str | None = None,
    semantic_proj_dim: int | None = None,
    semantic_candidate_k: int = 1200,
    lexical_rrf_weight: float = 1.0,
    semantic_rrf_weight: float | None = None,
    max_context_chars: int = 24000,
) -> dict:
    """Retrieve through the top-level LangChain LCEL pipeline."""
    request = dict(locals())
    from .pipeline import retrieve
    return retrieve(request)
