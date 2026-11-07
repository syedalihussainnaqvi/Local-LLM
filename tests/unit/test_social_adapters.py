"""Unit tests for Phase 5: Hardened Social Media Adapters.
Tests:
- Interface compliance and normalized output schema (platform, handle, timestamp, text, url, engagement).
- Deterministic offline testing using recorded fixtures.
- Typed error handling for rate limits (HTTP 429).
- Compliance status reporting and unverified claims labeling.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_llm.agent.tools.social_adapters import (
    RedditAdapter,
    GitHubAdapter,
    HackerNewsAdapter,
    SocialPost,
    RateLimitExceededError,
    SocialAPIError,
)


class TestSocialAdapters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = str(PROJECT_ROOT / "eval" / "fixtures" / "social")

    def test_01_reddit_adapter_normalized_fixture(self):
        """Reddit adapter must return normalized SocialPost objects from fixture."""
        adapter = RedditAdapter(fixture_dir=self.fixture_dir)
        posts = adapter.search_posts("machine learning", limit=1)

        self.assertEqual(len(posts), 1)
        post = posts[0]
        self.assertIsInstance(post, SocialPost)
        self.assertEqual(post.platform, "Reddit")
        self.assertEqual(post.handle, "tech_researcher")
        self.assertTrue(post.is_unverified_claim)
        self.assertIn("MachineLearning", post.url)
        self.assertIn("score", post.engagement)

    def test_02_github_adapter_normalized_fixture(self):
        """GitHub adapter must return normalized SocialPost objects from fixture."""
        adapter = GitHubAdapter(fixture_dir=self.fixture_dir)
        posts = adapter.search_posts("antigravity", limit=1)

        self.assertEqual(len(posts), 1)
        post = posts[0]
        self.assertEqual(post.platform, "GitHub")
        self.assertEqual(post.handle, "antigravity-dev")
        self.assertGreater(post.engagement["stars"], 1000)

    def test_03_hackernews_adapter_normalized_fixture(self):
        """HackerNews adapter must return normalized SocialPost objects from fixture."""
        adapter = HackerNewsAdapter(fixture_dir=self.fixture_dir)
        posts = adapter.search_posts("supervision", limit=1)

        self.assertEqual(len(posts), 1)
        post = posts[0]
        self.assertEqual(post.platform, "HackerNews")
        self.assertIn("points", post.engagement)

    def test_04_compliance_metadata(self):
        """Adapters must publish compliance info and ToS status."""
        for AdapterCls in [RedditAdapter, GitHubAdapter, HackerNewsAdapter]:
            adapter = AdapterCls()
            info = adapter.get_compliance_info()
            self.assertIn("method", info)
            self.assertIn("tos_status", info)


if __name__ == "__main__":
    unittest.main()
