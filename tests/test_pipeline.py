"""LangChain is the public orchestration layer, including optional generation."""
import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag import build_index, create_answer_chain, create_retrieval_chain
from rag.evaluation import evaluate

FIXTURE = Path(__file__).parent / 'fixtures' / 'corpus'


class LangChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.index = Path(cls.temp.name) / 'index'
        build_index(FIXTURE, cls.index, include_pdfs=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def request(self, query='How do loop invariants work?'):
        return {'index_dir': self.index, 'query_text': query, 'top_k': 4}

    def test_documents_citations_and_batch(self):
        chain = create_retrieval_chain()
        states = chain.batch([self.request(), self.request('integer overflow upper bound')])
        self.assertEqual(len(states), 2)
        self.assertTrue(all(isinstance(doc, Document) for doc in states[0]['documents']))
        self.assertIn('invariant', states[0]['context'])
        self.assertEqual(states[0]['documents'][0].metadata['index_generation'],
                         states[0]['retrieval']['index_generation'])
        self.assertIn('arithmetic.md', states[1]['documents'][0].metadata['path'])

    def test_async(self):
        state = asyncio.run(create_retrieval_chain().ainvoke(self.request()))
        self.assertTrue(state['documents'])

    def test_answer_chain_injects_evidence_and_input(self):
        def fake_chat(prompt):
            messages = prompt.to_messages()
            self.assertIn('[Evidence ', messages[0].content)
            self.assertIn('loop invariants', messages[1].content)
            self.assertIn('invariant not satisfied', messages[1].content)
            return AIMessage(content='Keep the invariant true before and after each iteration.')
        request = {**self.request(), 'error_text': 'invariant not satisfied'}
        state = create_answer_chain(RunnableLambda(fake_chat)).invoke(request)
        self.assertIn('before and after', state['answer'])
        self.assertTrue(state['documents'])

    def test_cli_and_evaluation_use_same_pipeline(self):
        output = subprocess.check_output([sys.executable, 'rag_cli.py', 'query',
            '--index-dir', str(self.index), '--query-text', 'loop invariant',
            '--semantic-backend', 'none'], text=True)
        self.assertIn('invariant', json.loads(output)['prompt_pack'])
        cases = Path(self.temp.name) / 'cases.jsonl'
        cases.write_text(json.dumps({'id': 'loop', 'query': 'loop invariant',
                                     'expected_paths': ['tutorial/loops.md']}) + '\n')
        report = evaluate(str(self.index), str(cases), semantic_backend='none')
        self.assertEqual(report['passed'], 1)
        self.assertEqual(report['hit_rate'], 1)


if __name__ == '__main__':
    unittest.main()
