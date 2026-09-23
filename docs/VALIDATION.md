# Local validation of the first refactor

The top-level framework is LangChain LCEL. This run used Python 3.12 and the
512-dimensional deterministic projection backend (not a learned embedding model).

- Editable package installation and the installed `verag` entry point passed.
- `python -m unittest discover -s tests -v`: **22 tests passed**.
- Full-corpus document-target smoke evaluation: **6/6 passed at top-5**.
- Index: 90,211 chunks from 876 Rust files, 4,128 tutorial files and 2 PDFs.
- Snapshot: `6a0692b7839b404fb3f3a97b2063a26e`.

| English query case | First target document rank |
| --- | --- |
| loop-basics | 1 |
| loop-exit | 2 |
| loop-error | 1 |
| loop-isolation | 5 |
| integer-types | 1 |
| mutable-reference | 3 |

Reproduce:

```sh
python -m pip install -e '.[pdf]'
python -m unittest discover -s tests -v
verag build --index-dir .rag_index
verag evaluate --index-dir .rag_index --output .rag_eval/smoke.json
verag query --query-text "How do loop invariants establish a postcondition after a while loop?" --top-k 5
```

The last example ranked `tutorial/verus/while.md` first. Detailed local JSON and
full-evidence context are in ignored `.rag_eval/` outputs, not committed artifacts.

Tests cover LCEL batch/async execution, fake-model answer generation, loop/error
retrieval, generics/formatting, code fences, include provenance, independent vector
recall, full evidence packing, rebuild cache invalidation, failed-build rollback,
legacy indexes, PDF page selection and explicit model/schema mismatch errors.

These six queries are development smoke checks used during the refactor, not an
independent held-out benchmark. Target paths are not exhaustive relevance labels;
no proof repair success rate or learned-model quality is claimed. The answer chain
was checked with a fake model, not a paid model API or the Verus verifier. The
configured Python 3.10/3.12 GitHub Actions matrix has not been run remotely here.

Of 271 mdBook include references, 1 could be expanded locally; 270 resolve outside
the available source snapshot and remain explicitly unavailable. The source example
snapshot must be restored before those code examples can be indexed. Index builds
are still full scans; full AST parsing, incremental updates, token-based budgets
and verifier execution remain future work.
