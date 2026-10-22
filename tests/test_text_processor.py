"""Tests for TextProcessor and sentence-boundary chunking."""

import unittest
from text_processor import TextProcessor, process_and_chunk


class TestTextProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = TextProcessor(chunk_size=200, chunk_overlap=40)

    def test_clean_text(self):
        raw = "Hello   world!\r\n\r\n\r\n\r\nThis is a   test    line.\n"
        cleaned = self.processor.clean_text(raw)
        self.assertEqual(cleaned, "Hello world!\n\nThis is a test line.")

    def test_chunking_metadata(self):
        text = (
            "## Section 1: Transformers\n\n"
            "Transformers are deep learning architectures introduced in 2017. "
            "They rely heavily on self-attention mechanisms to process sequence data concurrently.\n\n"
            "## Section 2: LoRA\n\n"
            "Low-Rank Adaptation freezes pretrained model weights and injects trainable rank decomposition matrices. "
            "This greatly reduces the number of trainable parameters."
        )

        chunks = self.processor.chunk_document(
            text=text,
            source_url="https://example.com/ml-guide",
            title="Machine Learning Guide",
            retrieved_at="2026-10-20T12:00:00Z",
            doc_hash="abcdef1234567890",
        )

        self.assertGreater(len(chunks), 0)
        for i, ch in enumerate(chunks):
            self.assertTrue(ch["chunk_id"].startswith("abcdef123456_"))
            self.assertEqual(ch["source_url"], "https://example.com/ml-guide")
            self.assertEqual(ch["title"], "Machine Learning Guide")
            self.assertEqual(ch["retrieved_at"], "2026-10-20T12:00:00Z")
            self.assertTrue(len(ch["chunk_hash"]) == 64)  # SHA-256
            self.assertGreater(len(ch["content"]), 0)

    def test_sentence_boundary_preservation(self):
        # Sentences shouldn't be abruptly sliced mid-word
        sentences = [
            "First sentence explains concept A thoroughly.",
            "Second sentence explains concept B with details.",
            "Third sentence concludes the paragraph neatly."
        ]
        text = " ".join(sentences)

        chunks = self.processor.chunk_document(
            text=text,
            source_url="https://example.com/test",
            title="Boundary Test"
        )

        for ch in chunks:
            # Each chunk should end with punctuation or paragraph terminal
            content = ch["content"].strip()
            self.assertTrue(content.endswith((".", "!", "?", "neatly.")))

    def test_empty_text_returns_empty_list(self):
        chunks = self.processor.chunk_document("", "https://example.com")
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
