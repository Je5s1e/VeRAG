# Current retrieval design

The top-level framework is LangChain LCEL (`rag/pipeline.py`). The request chain
validates input, calls Verus-specific retrieval, produces LangChain Documents and
builds the evidence context. `query_index()`, `verag` and evaluation are adapters
to that same chain. An optional answer chain composes a caller-provided chat model
with a grounded prompt and output parser. No HTTP server or model API is required
for retrieval. The chain supports invoke, batch, ainvoke and RunnableConfig callbacks.

## Data and persistence

`index_builder.py` reads project code, documentation and PDFs. `chunking.py`
preserves function regions and Markdown code blocks. The lexical Rust chunker
is conservative and explicitly not an AST parser. `includes.py` expands locally
available mdBook examples and records missing references without guessing content.

Schema v2 snapshots contain chunks, persisted lexical term counts and optional
float32 vectors. A build writes a new generation and publishes the manifest through
atomic rename. Readers pin one generation for the entire request; caches are keyed
by snapshot path. Historical root-level JSONL indexes are still readable.

IDs derive from path, source span, page and exact content hash. Include expansions
carry their own source span. Markdown citations refer to original document lines;
expanded text additionally cites the included source. PDF citations refer to pages.

## Query path

Code specs and normalized errors augment English query terms. BM25 and vector
similarity each retrieve independently. Document vectors are built offline; the
query uses the index's recorded embedding configuration. Projection embeddings
are deterministic feature hashing, not a trained language model. Learned models
are optional and must already be installed locally.

RRF combines channel ranks; output score has the same meaning as ranking. Explicit
zero weights disable a channel's contribution. Selection deduplicates overlapping
spans and near-identical full text, with a per-project-repo limit. Source minimums
are opt-in. `grouped` is diagnostic and may differ from selected results.

The context renderer packs complete formatted evidence blocks, including stable
IDs and citations. `snippet` is a short display preview only. Budgets are characters,
not tokens; large evidence can be omitted and remains in full result JSON.

## Verification

`tests/` provides fast offline behavior regressions. `evaluation/smoke_cases.jsonl`
provides document-target checks against the full local corpus. Reports include hit
rate and first-hit reciprocal rank, not exhaustive relevance or repair success.
The historical retrieval JSON reports are retained as historical data only.

## Remaining work

- Full Verus parser integration, parent/child proof blocks and dependency expansion.
- Incremental updates and persisted inverted/ANN indexes at larger scales.
- Task-aware relevance calibration, token budgets and optional reranking.
- Independently annotated held-out evaluation, HTTP service and Verus repair loop.

The architecture roadmap is in [docs/RAG_REFACTOR_PLAN.md](../docs/RAG_REFACTOR_PLAN.md).
It describes future work as well as this first implementation increment.
