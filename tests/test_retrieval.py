"""Tests for embeddings and HybridRetriever."""

import os
import tempfile
import unittest
import numpy as np

from local_llm.embeddings import LocalTextEmbedder
from local_llm.retrieval import HybridRetriever, format_context_for_prompt
from local_llm.storage import StorageManager


class TestRetrieval(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_rag.db")
        self.storage = StorageManager(db_path=self.db_path)
        self.embedder = LocalTextEmbedder(vector_dim=256)
        self.retriever = HybridRetriever(storage=self.storage, embedder=self.embedder)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_local_text_embedder(self):
        vec1 = self.embedder.embed_text("Deep neural networks and backpropagation")
        vec2 = self.embedder.embed_text("Neural networks training with gradient descent")
        vec3 = self.embedder.embed_text("Baking chocolate chip cookies in an oven")

        self.assertEqual(vec1.shape, (256,))
        # Normalized vectors (norm = 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(vec1)), 1.0, places=4)

        # Dot product = cosine similarity
        sim_ml = float(np.dot(vec1, vec2))
        sim_food = float(np.dot(vec1, vec3))
        self.assertGreater(sim_ml, sim_food)

    def test_hybrid_search_retrieval(self):
        url = "https://example.com/ai"
        chunks = [
            {
                "chunk_id": "c_ai_1",
                "source_url": url,
                "title": "AI & Transformers",
                "section": "Attention",
                "content": "Self-attention enables transformer models to dynamically weight different token positions.",
                "chunk_hash": "hash_ai_1",
                "retrieved_at": "2026-10-20T12:00:00Z",
                "char_count": 80,
            },
            {
                "chunk_id": "c_culinary_1",
                "source_url": "https://example.com/cooking",
                "title": "Cooking Italian Pasta",
                "section": "Sauces",
                "content": "Boil the pasta in salted water until al dente, then toss with olive oil and garlic sauce.",
                "chunk_hash": "hash_cook_1",
                "retrieved_at": "2026-10-20T12:00:00Z",
                "char_count": 80,
            },
        ]

        self.storage.upsert_document(
            url=url,
            title="AI & Transformers",
            text="AI text",
            content_hash="h1",
            retrieved_at="2026-10-20T12:00:00Z",
            chunks=[chunks[0]],
        )
        self.storage.upsert_document(
            url="https://example.com/cooking",
            title="Cooking Italian Pasta",
            text="Cooking text",
            content_hash="h2",
            retrieved_at="2026-10-20T12:00:00Z",
            chunks=[chunks[1]],
        )

        self.retriever.refresh_index()

        # Query about attention mechanisms
        results = self.retriever.search("How does self-attention work in transformers?", top_k=2)
        self.assertGreater(len(results), 0)
        top_result = results[0]
        self.assertEqual(top_result["chunk_id"], "c_ai_1")
        self.assertIn("Self-attention enables", top_result["content"])
        self.assertGreater(top_result["score"], 0.1)

    def test_format_context_for_prompt(self):
        mock_chunks = [
            {
                "source_url": "https://example.com/doc",
                "title": "Documentation",
                "section": "Overview",
                "content": "This is documented evidence.",
                "retrieved_at": "2026-10-20T12:00:00Z",
                "score": 0.85,
            }
        ]

        formatted_text, citations = format_context_for_prompt(mock_chunks, max_chars=1000)
        self.assertIn("[Source 1: Documentation] (https://example.com/doc)", formatted_text)
        self.assertIn("This is documented evidence.", formatted_text)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["url"], "https://example.com/doc")


if __name__ == "__main__":
    unittest.main()
