"""Tests for StorageManager SQLite persistence and incremental updates."""

import os
import tempfile
import unittest

from storage import StorageManager


class TestStorage(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_storage.db")
        self.storage = StorageManager(db_path=self.db_path)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_upsert_and_retrieve_document(self):
        url = "https://example.com/page1"
        title = "Page 1 Title"
        text = "This is the full text of page 1."
        doc_hash = "hash1111111111111111"
        chunks = [
            {
                "chunk_id": "chunk_001",
                "source_url": url,
                "title": title,
                "section": "Intro",
                "content": "This is chunk 1.",
                "chunk_hash": "chash1",
                "retrieved_at": "2026-10-20T12:00:00Z",
                "char_count": 16,
            }
        ]

        is_new, msg = self.storage.upsert_document(
            url=url,
            title=title,
            text=text,
            content_hash=doc_hash,
            retrieved_at="2026-10-20T12:00:00Z",
            chunks=chunks,
        )

        self.assertTrue(is_new)
        self.assertIn("version 1", msg)

        # Retrieve source
        source = self.storage.get_source(url)
        self.assertIsNotNone(source)
        self.assertEqual(source["title"], title)
        self.assertEqual(source["version"], 1)

        # Retrieve chunks
        all_chunks = self.storage.get_chunks_for_url(url)
        self.assertEqual(len(all_chunks), 1)
        self.assertEqual(all_chunks[0]["content"], "This is chunk 1.")

    def test_incremental_update_skip_unchanged(self):
        url = "https://example.com/page2"
        doc_hash = "same_hash_123456"

        # First insert
        self.storage.upsert_document(
            url=url,
            title="Static Page",
            text="Static content",
            content_hash=doc_hash,
            retrieved_at="2026-10-20T12:00:00Z",
            chunks=[{"chunk_id": "c1", "source_url": url, "title": "Static Page", "content": "Static", "chunk_hash": "ch1"}],
        )

        # Refresh with identical hash
        is_new, msg = self.storage.upsert_document(
            url=url,
            title="Static Page",
            text="Static content",
            content_hash=doc_hash,
            retrieved_at="2026-10-20T13:00:00Z",
            chunks=[],
        )

        self.assertFalse(is_new)
        self.assertIn("unchanged", msg.lower())

        # Version should still be 1
        source = self.storage.get_source(url)
        self.assertEqual(source["version"], 1)

    def test_update_changed_content_increments_version(self):
        url = "https://example.com/page3"

        # Version 1
        self.storage.upsert_document(
            url=url,
            title="V1 Title",
            text="Version 1 text",
            content_hash="hash_v1",
            retrieved_at="2026-10-20T12:00:00Z",
            chunks=[{"chunk_id": "c1", "source_url": url, "title": "V1", "content": "V1", "chunk_hash": "h1"}],
        )

        # Version 2 with new content hash
        is_new, msg = self.storage.upsert_document(
            url=url,
            title="V2 Title",
            text="Version 2 text updated",
            content_hash="hash_v2",
            retrieved_at="2026-10-20T14:00:00Z",
            chunks=[{"chunk_id": "c2", "source_url": url, "title": "V2", "content": "V2 updated", "chunk_hash": "h2"}],
        )

        self.assertTrue(is_new)
        self.assertIn("version 2", msg.lower())

        source = self.storage.get_source(url)
        self.assertEqual(source["version"], 2)
        self.assertEqual(source["title"], "V2 Title")

        # Chunks should be replaced
        chunks = self.storage.get_chunks_for_url(url)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "c2")

    def test_delete_cascades_to_chunks(self):
        url = "https://example.com/delete_me"
        self.storage.upsert_document(
            url=url,
            title="To Delete",
            text="Content",
            content_hash="h1",
            retrieved_at="2026-10-20T12:00:00Z",
            chunks=[{"chunk_id": "del_1", "source_url": url, "title": "To Delete", "content": "Content", "chunk_hash": "h1"}],
        )

        self.assertTrue(self.storage.delete_source(url))
        self.assertIsNone(self.storage.get_source(url))
        self.assertEqual(len(self.storage.get_chunks_for_url(url)), 0)

    def test_crawl_history_logging(self):
        self.storage.record_history("https://example.com", "scrape", "success", "Fetched 500 chars")
        history = self.storage.get_history(limit=10)
        self.assertGreater(len(history), 0)
        self.assertEqual(history[0]["action"], "scrape")
        self.assertEqual(history[0]["status"], "success")


if __name__ == "__main__":
    unittest.main()
