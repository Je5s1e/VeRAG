"""Reproducible document-target smoke evaluations (not relevance judgments)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .retriever import query_index


def evaluate(index_dir: str, cases_path: str, top_k: int = 5, semantic_backend: str = 'auto') -> dict:
    cases = [json.loads(line) for line in Path(cases_path).read_text().splitlines() if line.strip()]
    if not cases:
        raise ValueError('Evaluation cases must not be empty')
    reports = []
    for case in cases:
        expected = case['expected_paths']
        if not expected:
            raise ValueError(f"Case {case['id']} has no expected paths")
        start = time.perf_counter()
        result = query_index(index_dir, case.get('query', ''), code_text=case.get('code', ''),
                             error_text=case.get('error', ''), top_k=top_k,
                             semantic_backend=semantic_backend)
        elapsed = (time.perf_counter() - start) * 1000
        paths = [item['path'] for item in result['results']]
        rank = next((i for i, path in enumerate(paths, 1) if path in expected), None)
        reports.append({'id': case['id'], 'query': case.get('query', ''), 'expected_paths': expected,
                        'paths': paths, 'first_relevant_rank': rank, 'passed': rank is not None,
                        'latency_ms': round(elapsed, 2), 'debug': result['retrieval_debug']})
    return {'evaluation_type': 'document_target_smoke', 'top_k': top_k,
            'index_generation': result['index_generation'], 'backend': semantic_backend,
            'case_count': len(reports), 'passed': sum(x['passed'] for x in reports),
            'hit_rate': sum(x['passed'] for x in reports) / len(reports),
            'mrr': sum(1 / x['first_relevant_rank'] if x['first_relevant_rank'] else 0 for x in reports) / len(reports),
            'cases': reports}
