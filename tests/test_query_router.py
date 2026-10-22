"""Tests for QueryRouter query intent detection and mode routing."""

import os
import tempfile
import unittest

from query_router import QueryRouter
from retrieval import HybridRetriever
from storage import StorageManager


class TestQueryRouter(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_router.db")
        self.storage = StorageManager(db_path=self.db_path)
        self.retriever = HybridRetriever(storage=self.storage)
        self.router = QueryRouter(storage=self.storage, retriever=self.retriever)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_direct_url_detection(self):
        prompt = "Can you extract insights from https://en.wikipedia.org/wiki/LoRA ?"
        res = self.router.route_query(prompt, user_override_mode="auto")
        self.assertEqual(res["route"], "direct_url")
        self.assertIn("https://en.wikipedia.org/wiki/LoRA", res["extracted_urls"])

    def test_live_web_keyword_routing(self):
        prompt = "What are the latest breakthrough AI models released this month?"
        res = self.router.route_query(prompt, user_override_mode="auto", web_access_enabled=True)
        self.assertEqual(res["route"], "live_web")

    def test_live_web_fallback_when_web_disabled(self):
        prompt = "What are the latest breakthrough AI models released this month?"
        res = self.router.route_query(prompt, user_override_mode="auto", web_access_enabled=False)
        # Should NOT route to live_web when web access is disabled
        self.assertNotEqual(res["route"], "live_web")

    def test_explicit_user_mode_override(self):
        prompt = "What is the capital of France?"
        # Explicit local_rag
        res_rag = self.router.route_query(prompt, user_override_mode="local_rag")
        self.assertEqual(res_rag["route"], "local_rag")

        # Explicit llm_only
        res_llm = self.router.route_query("latest breaking news", user_override_mode="llm_only")
        self.assertEqual(res_llm["route"], "llm_only")

    def test_greeting_routes_to_llm_only(self):
        prompt = "Hello!"
        res = self.router.route_query(prompt, user_override_mode="auto", web_access_enabled=True)
        self.assertEqual(res["route"], "llm_only")

    def test_question_automatic_web_scraping(self):
        prompt = "Explain how quantum computers use qubits for computation."
        res = self.router.route_query(prompt, user_override_mode="auto", web_access_enabled=True)
        self.assertEqual(res["route"], "live_web")

    def test_general_question_web_disabled(self):
        prompt = "Explain how backpropagation calculates gradients in neural networks."
        res = self.router.route_query(prompt, user_override_mode="auto", web_access_enabled=False)
        self.assertEqual(res["route"], "llm_only")


if __name__ == "__main__":
    unittest.main()
