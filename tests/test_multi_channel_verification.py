"""Tests for multi-channel verification of news, sports, and Google research routing.

Live-network tests are replaced with mock-based offline equivalents so the suite
passes without an internet connection (H-4: no implicit live dependencies in CI).
"""

import unittest
from unittest.mock import patch, MagicMock
from local_llm.web_discovery import is_news_query, is_sports_query, CompositeAuthenticProvider, TOP_NEWS_CHANNELS, TOP_SPORTS_CHANNELS


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

    @patch("local_llm.web_discovery.DuckDuckGoLiteProvider.search", return_value=[])
    @patch("local_llm.web_discovery.GoogleNewsRssProvider.search_items")
    def test_news_search_groups_distinct_channels(self, mock_items, _mock_ddg):
        """Offline: patch RSS feed; verify category tagging and dedup logic.
        Provides 5 distinct-channel items so the provider's early-return fires
        before falling through to the DDG research path.
        """
        mock_items.return_value = [
            {"url": "https://bbc.com/1",     "title": "World News 1", "snippet": "s1", "channel": "BBC",      "source": "BBC"},
            {"url": "https://reuters.com/1", "title": "World News 2", "snippet": "s2", "channel": "Reuters",  "source": "Reuters"},
            {"url": "https://cnn.com/1",     "title": "World News 3", "snippet": "s3", "channel": "CNN",      "source": "CNN"},
            {"url": "https://aljazeera.com/1","title": "World News 4","snippet": "s4", "channel": "Al Jazeera","source": "Al Jazeera"},
            {"url": "https://apnews.com/1",  "title": "World News 5", "snippet": "s5", "channel": "AP News",  "source": "AP News"},
        ]
        provider = CompositeAuthenticProvider()
        results = provider.search("latest world news", max_results=5)
        self.assertGreaterEqual(len(results), 1)
        for r in results:
            self.assertIn("category", r)
            self.assertEqual(r["category"], "news")

    @patch("local_llm.web_discovery.DuckDuckGoLiteProvider.search", return_value=[])
    @patch("local_llm.web_discovery.GoogleNewsRssProvider.search_items")
    def test_sports_search_groups_distinct_channels(self, mock_items, _mock_ddg):
        """Offline: patch RSS feed with 5 distinct-channel sports fixture items."""
        mock_items.return_value = [
            {"url": "https://espn.com/1",      "title": "Cricket World Cup Update",  "snippet": "Pakistan vs India",    "channel": "ESPN",       "source": "ESPN"},
            {"url": "https://cricinfo.com/1",  "title": "World Cup Semi Finals",     "snippet": "Semi final results",  "channel": "Cricinfo",   "source": "Cricinfo"},
            {"url": "https://skysports.com/1", "title": "World Cup Day 3",           "snippet": "Day 3 highlights",    "channel": "Sky Sports", "source": "Sky Sports"},
            {"url": "https://bbc.com/sport/1", "title": "BBC Sports Roundup",        "snippet": "Match roundup",      "channel": "BBC Sport",  "source": "BBC Sport"},
            {"url": "https://sport24.com/1",   "title": "World Cup Final Preview",   "snippet": "Final preview",      "channel": "Sport24",    "source": "Sport24"},
        ]
        provider = CompositeAuthenticProvider()
        results = provider.search("cricket world cup", max_results=5)
        self.assertGreaterEqual(len(results), 1)
        for r in results:
            self.assertIn("category", r)
            self.assertEqual(r["category"], "sports")

    @patch("local_llm.web_discovery.DuckDuckGoLiteProvider.search")
    @patch("local_llm.web_discovery.WikipediaProvider.search")
    def test_general_research_routing(self, mock_wiki, mock_ddg):
        """Offline: patch DDG + Wikipedia; verify composite result merging."""
        mock_ddg.return_value = [
            {"url": "https://ddg.com/1", "title": "Photosynthesis explained", "snippet": "Plants use sunlight.", "source": "DuckDuckGo"},
        ]
        mock_wiki.return_value = [
            {"url": "https://en.wikipedia.org/wiki/Photosynthesis", "title": "Photosynthesis", "snippet": "Biological process.", "source": "Wikipedia"},
        ]
        provider = CompositeAuthenticProvider()
        results = provider.search("photosynthesis process explanation", max_results=3)
        self.assertGreaterEqual(len(results), 1)

    def test_keyword_research_query_detection(self):
        from local_llm.web_discovery import is_keyword_research_query, extract_target_keyword
        self.assertTrue(is_keyword_research_query("find from the google or any keyword search tool the search volume and KD of the produces and reviews"))
        self.assertTrue(is_keyword_research_query("search volume of iphone 16"))
        self.assertTrue(is_keyword_research_query("what is the keyword difficulty of python"))
        self.assertFalse(is_keyword_research_query("how to write a python loop"))
        self.assertEqual(extract_target_keyword("find from the google or any keyword search tool the search volume and KD of the produces and reviews"), "produces and reviews")

    def test_keyword_research_provider(self):
        """Offline: keyword research provider returns deterministic synthetic data (no HTTP)."""
        from local_llm.web_discovery import CompositeAuthenticProvider
        provider = CompositeAuthenticProvider()
        # GoogleKeywordMetricsProvider generates synthetic results without network calls
        results = provider.search("find search volume and KD of the produces and reviews", max_results=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].get("category"), "keyword_research")
        self.assertIn("produces and reviews", results[0].get("title", "").lower())
        self.assertIn("Search Volume", results[0].get("snippet", ""))


if __name__ == "__main__":
    unittest.main()
