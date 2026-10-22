"""Tests for multi-channel verification of news, sports, and Google research routing."""

import unittest
from web_discovery import is_news_query, is_sports_query, CompositeAuthenticProvider, TOP_NEWS_CHANNELS, TOP_SPORTS_CHANNELS


class TestMultiChannelVerification(unittest.TestCase):

    def test_news_query_detection(self):
        self.assertTrue(is_news_query("what is today's breaking news"))
        self.assertTrue(is_news_query("latest election headlines in pakistan"))
        self.assertTrue(is_news_query("what happened in middle east ceasefire"))
        self.assertFalse(is_news_query("what is quantum computing"))

    def test_sports_query_detection(self):
        self.assertTrue(is_sports_query("who won the cricket match today"))
        self.assertTrue(is_sports_query("latest champions league football score"))
        self.assertTrue(is_sports_query("babar azam batting stats"))
        self.assertFalse(is_sports_query("how to bake chocolate cake"))

    def test_top_channels_lists(self):
        self.assertIn("BBC", TOP_NEWS_CHANNELS)
        self.assertIn("Reuters", TOP_NEWS_CHANNELS)
        self.assertIn("CNN", TOP_NEWS_CHANNELS)
        self.assertIn("ESPN", TOP_SPORTS_CHANNELS)
        self.assertIn("Sky Sports", TOP_SPORTS_CHANNELS)
        self.assertIn("Cricinfo", TOP_SPORTS_CHANNELS)

    def test_news_search_groups_distinct_channels(self):
        provider = CompositeAuthenticProvider()
        # Mock or live test
        results = provider.search("latest world news", max_results=5)
        self.assertGreaterEqual(len(results), 1)
        for r in results:
            self.assertIn("category", r)
            self.assertEqual(r["category"], "news")

    def test_sports_search_groups_distinct_channels(self):
        provider = CompositeAuthenticProvider()
        results = provider.search("cricket world cup", max_results=5)
        self.assertGreaterEqual(len(results), 1)
        for r in results:
            self.assertIn("category", r)
            self.assertEqual(r["category"], "sports")

    def test_general_research_routing(self):
        provider = CompositeAuthenticProvider()
        results = provider.search("photosynthesis process explanation", max_results=3)
        self.assertGreaterEqual(len(results), 1)
    def test_keyword_research_query_detection(self):
        from web_discovery import is_keyword_research_query, extract_target_keyword
        self.assertTrue(is_keyword_research_query("find from the google or any keyword search tool the search volume and KD of the produces and reviews"))
        self.assertTrue(is_keyword_research_query("search volume of iphone 16"))
        self.assertTrue(is_keyword_research_query("what is the keyword difficulty of python"))
        self.assertFalse(is_keyword_research_query("how to write a python loop"))
        self.assertEqual(extract_target_keyword("find from the google or any keyword search tool the search volume and KD of the produces and reviews"), "produces and reviews")

    def test_keyword_research_provider(self):
        from web_discovery import CompositeAuthenticProvider
        provider = CompositeAuthenticProvider()
        results = provider.search("find search volume and KD of the produces and reviews", max_results=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].get("category"), "keyword_research")
        self.assertIn("produces and reviews", results[0].get("title", "").lower())
        self.assertIn("Search Volume", results[0].get("snippet", ""))


if __name__ == "__main__":
    unittest.main()
