"""Offline behavioral regressions: a real miniature corpus, no model downloads."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rag import build_index, query_index, render_prompt_context
from rag.chunking import code_chunks, document_chunks
from rag.code_analyzer import _extract_spec_clauses
from rag.index_builder import _clean_doc_text
from rag.storage import resolve_snapshot

FIXTURE = Path(__file__).parent / 'fixtures' / 'corpus'


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'corpus'
        shutil.copytree(FIXTURE, self.root)
        self.index = Path(self.temp.name) / 'index'
        self.meta = build_index(self.root, self.index, include_pdfs=False)

    def query(self, text, **kwargs):
        return query_index(self.index, text, **kwargs)

    def test_loop_invariant_query_retrieves_explanation_and_code(self):
        result = self.query('How do loop invariants establish a postcondition after a while loop?', top_k=4)
        paths = [x['path'] for x in result['results']]
        self.assertIn('tutorial/loops.md', paths)
        self.assertIn('projects/demo/count.rs', paths)
        self.assertIn('invariant i <= n', result['prompt_pack'])
        self.assertNotIn('{{#include', result['prompt_pack'])

    def test_error_only_retrieval(self):
        result = self.query('', error_text='error: invariant not satisfied at end of loop body')
        self.assertIn('invariant', result['error_categories'])
        self.assertTrue(any(x['path'] == 'tutorial/loops.md' for x in result['results'][:3]))

    def test_other_topics_and_no_match(self):
        for query, expected in [('integer overflow arithmetic upper bound', 'arithmetic.md'),
                                ('forall quantifier trigger instantiation', 'triggers.md'),
                                ('install Rust toolchain release', 'install.md')]:
            result = self.query(query, semantic_backend='none', top_k=2)
            self.assertTrue(any(x['path'].endswith(expected) for x in result['results']), query)
        self.assertEqual(self.query('zzzz_nonexistent_symbol', semantic_backend='none')['results'], [])
        self.assertEqual(self.query('')['results'], [])

    def test_generation_rebuild_invalidates_warm_cache(self):
        self.query('loop invariant')
        (self.root / 'tutorial' / 'new.md').write_text('# New\nuniquesentinel lemma')
        meta = build_index(self.root, self.index, include_pdfs=False)
        result = self.query('uniquesentinel', semantic_backend='none')
        self.assertNotEqual(self.meta['generation'], meta['generation'])
        self.assertEqual(result['results'][0]['path'], 'tutorial/new.md')
        self.assertEqual(result['index_generation'], meta['generation'])

    def test_ids_stable_and_failed_build_keeps_snapshot(self):
        before, _ = resolve_snapshot(self.index)
        original = (before / 'chunks.jsonl').read_text()
        build_index(self.root, self.index, include_pdfs=False)
        after, _ = resolve_snapshot(self.index)
        self.assertEqual(original, (after / 'chunks.jsonl').read_text())
        with patch('rag.index_builder.SemanticScorer.embed', side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):
                build_index(self.root, self.index, include_pdfs=False)
        self.assertEqual(resolve_snapshot(self.index)[0], after)

    def test_independent_dense_recall_and_cached_document_vectors(self):
        snapshot, _ = resolve_snapshot(self.index)
        chunks = [json.loads(line) for line in (snapshot / 'chunks.jsonl').read_text().splitlines()]
        target = next(i for i, c in enumerate(chunks) if c['path'] == 'tutorial/loops.md')
        vectors = np.zeros((len(chunks), 512), dtype=np.float32)
        vectors[target, 0] = 1
        # Controlled embeddings test channel independence, not model quality.
        np.save(snapshot / 'vectors.npy', vectors)
        query_vec = np.zeros((1, 512), dtype=np.float32)
        query_vec[0, 0] = 1
        with patch('rag.semantic.SemanticScorer.embed', return_value=query_vec) as embed:
            result = self.query('unseenqueryword')
        self.assertEqual(result['results'][0]['path'], 'tutorial/loops.md')
        self.assertEqual(result['retrieval_debug']['lexical_candidates'], 0)
        self.assertEqual(embed.call_count, 1)
        self.assertEqual(embed.call_args.args[0], ['unseenqueryword'])

    def test_full_evidence_and_context_budget(self):
        body = '# A long proof\n' + 'loop invariant explanatory evidence\n' * 25 + 'FINAL_PROOF_LINE\n'
        (self.root / 'tutorial' / 'long.md').write_text(body)
        build_index(self.root, self.index, include_pdfs=False)
        result = self.query('FINAL_PROOF_LINE', semantic_backend='none')
        self.assertNotIn('FINAL_PROOF_LINE', result['results'][0]['snippet'])
        self.assertIn('FINAL_PROOF_LINE', result['prompt_pack'])
        self.assertIn('\n', result['results'][0]['text'])
        self.assertEqual(render_prompt_context(result, max_chars=30), '')
        scores = [x['score'] for x in result['results']]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_validation_and_missing_index(self):
        with self.assertRaises(ValueError):
            self.query('loop', top_k=0)
        with self.assertRaises(ValueError):
            self.query('loop', semantic_backend='sentence-transformer')
        with self.assertRaises(FileNotFoundError):
            query_index(self.index / 'missing', 'loop')

    def test_include_provenance_and_source_lines(self):
        result = self.query('invariant', semantic_backend='none')
        example = next(x for x in result['results'] if any(r.get('status') == 'expanded' for r in x['includes']))
        self.assertEqual(example['includes'][0]['path'], 'references/count.rs')
        lines = (self.root / example['path']).read_text().splitlines()
        self.assertLessEqual(example['line_end'], len(lines))
        self.assertIn('references/count.rs', result['prompt_pack'])


class SourcePreservationTests(unittest.TestCase):
    def test_generic_types_and_indentation_survive(self):
        text = 'Seq<int> and Vec<u64>\n    assert(x < y);\n'
        self.assertEqual(_clean_doc_text(text), text)

    def test_long_function_not_split_by_window_or_fake_function_in_comment(self):
        text = 'fn long_proof() {\n/* fn fake() {} */\n' + '    assert(true);\n' * 100 + '}\nfn next() {}\n'
        chunks = code_chunks(text)
        self.assertEqual(len(chunks), 2)
        self.assertIn('assert(true);\n}', chunks[0][2])
        self.assertEqual(''.join(c[2] for c in chunks), text)

    def test_fenced_code_and_markdown_positions(self):
        text = '# Proof\n\n```rust\n' + '\nassert(true);' * 50 + '\n```\n\n# Next\nOther\n'
        chunks = document_chunks(text, target_chars=50)
        self.assertTrue(any('```rust' in c[2] and '\n```' in c[2] for c in chunks))
        for start, end, body in chunks:
            self.assertEqual(body, ''.join(text.splitlines(keepends=True)[start-1:end]))

    def test_quantifier_clause_block_survives(self):
        code = 'fn f() ensures forall|i: int| { i >= 0 ==> i + 1 > 0 }, { assert(true); }'
        clause = _extract_spec_clauses(code)['ensures']
        self.assertIn('i + 1 > 0', clause)
        self.assertNotIn('assert', clause)


if __name__ == '__main__':
    unittest.main()
