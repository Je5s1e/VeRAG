"""Persistence/fallback and include boundary regressions."""
import json
import tempfile
import unittest
from pathlib import Path

from rag import build_index, query_index
from rag.includes import expand_includes
from rag.storage import resolve_snapshot

FIXTURE = Path(__file__).parent / 'fixtures' / 'corpus'


class StorageAndIncludeTests(unittest.TestCase):
    def test_lexical_only_fallback_and_explicit_backend_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            build_index(FIXTURE, temp, include_pdfs=False, semantic_backend='none')
            result = query_index(temp, 'loop invariant')
            self.assertIn('lexical retrieval only', result['retrieval_debug']['degraded_reason'])
            with self.assertRaises(ValueError):
                query_index(temp, 'loop', semantic_backend='projection')

    def test_legacy_jsonl_and_different_pdf_pages(self):
        with tempfile.TemporaryDirectory() as temp:
            chunks = [dict(id=str(page), path='reference.pdf', page=page,
                           text=text, line_start=1, line_end=1, source_type='doc',
                           source_group='pdf', lang='pdf') for page, text in [
                (1, 'loop invariant must hold before the first iteration'),
                (2, 'loop exit uses the negated condition to derive a postcondition')]]
            Path(temp, 'chunks.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in chunks))
            result = query_index(temp, 'loop', semantic_backend='none')
            self.assertEqual({x['page'] for x in result['results']}, {1, 2})
            self.assertEqual(result['index_generation'], 'legacy')

    def test_schema_and_embedding_config_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            build_index(FIXTURE, temp, include_pdfs=False)
            with self.assertRaises(ValueError):
                query_index(temp, 'loop', semantic_proj_dim=7)
            with self.assertRaises(ValueError):
                query_index(temp, 'loop', semantic_model='wrong-model')
            manifest = Path(temp, 'meta.json')
            metadata = json.loads(manifest.read_text())
            metadata['schema_version'] = 999
            manifest.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                resolve_snapshot(temp)

    def test_empty_index_is_consistent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp, 'source')
            root.mkdir()
            index = Path(temp, 'index')
            build_index(root, index, include_pdfs=False)
            result = query_index(index, 'loop invariant')
            self.assertEqual(result['results'], [])
            self.assertEqual(result['prompt_pack'], '')
            self.assertIn('retrieval_debug', result)

    def test_missing_or_outside_include_is_not_fabricated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / 'guide.md'
            text, refs = expand_includes('{{#include missing.rs:example}}', source, root)
            self.assertIn('unavailable', text)
            self.assertEqual(refs[0]['status'], 'missing')
            text, refs = expand_includes('{{#include ../outside.rs}}', source, root)
            self.assertIn('outside', text)
            self.assertEqual(refs[0]['status'], 'outside_source_root')


if __name__ == '__main__':
    unittest.main()
