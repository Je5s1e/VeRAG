# Contributing

Use Python 3.10+ and a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[pdf]'
python -m unittest discover -s tests -v
```

Tests use a small committed corpus under `tests/fixtures/corpus`; they never
require a downloaded embedding model, an external API, or a Verus installation.
The Rust fixtures are retrieval inputs, not claims of independently verified code.

For retrieval changes, also build the full local corpus and run:

```sh
verag build --repo-root . --index-dir .rag_index
verag evaluate --index-dir .rag_index --output .rag_eval/smoke.json
```

Add behavioral regressions for changed parsing, ranking, citation or persistence
behavior. Do not put test questions or answers into the production corpus.
Keep source snapshots under `projects/` and `tutorial/` separate from package code.
Do not commit generated indexes, vectors, model weights or local reports.
Document changes to the index schema and CLI. Keep library and CLI interfaces
consistent and retain support for reading legacy JSONL indexes where practical.

Third-party projects retain their upstream licenses; do not relicense the vendored
corpus or copy it into test fixtures without preserving attribution. The root
package currently has no declared license; maintainers must choose one before
advertising a license grant. This refactor does not select one on their behalf.
