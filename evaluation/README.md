# Retrieval smoke cases

`smoke_cases.jsonl` contains English questions and expected document paths in the
vendored corpus. Run `verag evaluate` after building the index. A nonzero exit
code means at least one target document was absent from top-k.

The report includes document hit rate, first-hit reciprocal rank, candidate counts
and per-query wall time (the first query includes cold index loading). These are
small development smoke cases, not a held-out benchmark or a measurement of proof
repair success. Multiple chunks can reference the same document. Expected paths
are document targets, not exhaustive relevance labels; do not call this Recall@k.

Add independently annotated, project-separated cases before making broader quality
claims. Unit regressions and their miniature corpus live in `tests/`.
