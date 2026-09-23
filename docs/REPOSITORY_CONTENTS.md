# Published repository contents

VeRAG publishes its Python package, CLI, dependency metadata, offline tests and
fixtures, evaluation questions, GitHub Actions workflow, and user/developer docs.

The knowledge corpus is intentionally included so a clone can build and evaluate
an index without downloading additional repositories. Keep `projects/**/*.rs`,
`tutorial/**/*.md`, the two tutorial PDFs, and upstream source/configuration files.
Preserve upstream LICENSE, NOTICE, attribution and source notes. These snapshots
are retrieval inputs, not a guarantee that every vendored project can be built
standalone from this curated copy.

Do not publish generated indexes (`.rag_index`), local evaluation output
(`.rag_eval`), model weights, virtual environments, Python caches, package build
artifacts, credentials, or developer-specific logs. Historical JSON run reports
under `rag/` are local-only; maintained test cases live in `tests/` and `evaluation/`.

Generated Vest HTML/JavaScript/font documentation and upstream benchmark binary
inputs/domain lists are also excluded: VeRAG's loaders do not consume them. Text
license/source notes remain included. Removed tracked artifacts remain on the
maintainer's disk; `.gitignore` prevents accidentally adding them again.

This cleanup changes the current repository tree. Previously committed artifacts
remain in Git history; it does not rewrite or purge historical objects.

Upstream projects retain their own licenses. No root package license has been
chosen yet; do not infer a license grant for VeRAG's original code. Maintainers
should choose a license before presenting it as licensed open-source software.
