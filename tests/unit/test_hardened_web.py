"""Unit tests for Phase 4: Hardened Web Scraper, SSRF Defense, and Source Independence.
Tests:
- C-04: SSRF defenses (loopback, private subnets, cloud metadata, scheme restrictions).
- Section 4: Content extraction and HTML boilerplate stripping.
- Section 4: SimHash 64-bit shingling, wire-syndication detection, and >=2 independent source verification.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_llm.agent.tools.hardened_web import GuardedFetcher, ContentExtractor, IndependenceChecker, SSRFSecurityException


class TestHardenedWeb(unittest.TestCase):

    def test_01_ssrf_disallowed_schemes(self):
        """Disallowed schemes like file:// and ftp:// must be rejected."""
        safe, reason = GuardedFetcher.validate_url("file:///C:/Windows/System32/drivers/etc/hosts")
        self.assertFalse(safe)
        self.assertIn("disallowed scheme", reason.lower())

        safe2, _ = GuardedFetcher.validate_url("ftp://example.com/secret.zip")
        self.assertFalse(safe2)

    def test_02_ssrf_loopback_and_private_ips(self):
        """Loopback (127.0.0.1, localhost) and private IP ranges must be blocked."""
        blocked_targets = [
            "http://127.0.0.1:8080/admin",
            "http://localhost:3000",
            "http://192.168.1.1",
            "http://10.0.0.5:80",
            "http://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/computeMetadata/v1/",
        ]
        for url in blocked_targets:
            safe, reason = GuardedFetcher.validate_url(url)
            self.assertFalse(safe, f"Failed to block SSRF URL: {url} (reason: {reason})")

    def test_03_content_extractor(self):
        """Extractor must strip scripts, nav, footer and return clean text."""
        raw_html = """
        <html>
            <head><title>Quantum Computing Breakthrough</title></head>
            <body>
                <nav><a href="/">Home</a> | <a href="/about">About</a></nav>
                <script>alert("malicious script");</script>
                <style>body { color: red; }</style>
                <main>
                    <h1>Qubits Achieve Record Coherence</h1>
                    <p>Scientists have stabilized superconducting qubits for over 100 milliseconds.</p>
                </main>
                <footer>Copyright 2026 TechWire. All rights reserved.</footer>
            </body>
        </html>
        """
        extracted = ContentExtractor.extract(raw_html, "https://example-news.com/quantum")
        self.assertEqual(extracted["title"], "Quantum Computing Breakthrough")
        self.assertIn("Qubits Achieve Record Coherence", extracted["clean_text"])
        self.assertIn("superconducting qubits", extracted["clean_text"])
        self.assertNotIn("malicious script", extracted["clean_text"])
        self.assertNotIn("Copyright 2026", extracted["clean_text"])

    def test_04_simhash_syndicated_articles_detection(self):
        """Syndicated wire copies must cluster together (1 independent source)."""
        wire_article_a = (
            "Federal regulators today announced updated autonomous vehicle safety standards. "
            "The new framework establishes strict guidelines for real-time sensor verification and fallback braking systems."
        )
        # Wire copy with minor punctuation and header change
        wire_article_b = (
            "Breaking: Federal regulators today announced updated autonomous vehicle safety standards. "
            "The framework establishes strict guidelines for real-time sensor verification and fallback braking systems."
        )

        h1 = IndependenceChecker._simhash(wire_article_a)
        h2 = IndependenceChecker._simhash(wire_article_b)
        dist = IndependenceChecker.hamming_distance(h1, h2)
        self.assertLessEqual(dist, 14, f"Expected syndicated articles to have small Hamming distance, got {dist}")

        articles = [
            {"clean_text": wire_article_a, "domain": "wire-reuters.com"},
            {"clean_text": wire_article_b, "domain": "news-mirror.com"},
        ]
        res = IndependenceChecker.verify_independence(articles, min_independent=2)
        # Both are syndicated copies of the same wire story -> only 1 independent cluster
        self.assertEqual(res["independent_source_count"], 1)
        self.assertFalse(res["verified"], "Syndicated wire reprints should not count as 2 independent sources!")

    def test_05_independent_sources_verification(self):
        """Genuinely distinct articles must yield >=2 independent clusters."""
        article_tech = (
            "Quantum processors have demonstrated error mitigation using surface codes in university lab trials."
        )
        article_policy = (
            "The central bank lowered benchmark interest rates by 25 basis points following the quarterly inflation report."
        )

        articles = [
            {"clean_text": article_tech, "domain": "tech-journal.org"},
            {"clean_text": article_policy, "domain": "financial-times.org"},
        ]
        res = IndependenceChecker.verify_independence(articles, min_independent=2)
        self.assertEqual(res["independent_source_count"], 2)
        self.assertTrue(res["verified"])


if __name__ == "__main__":
    unittest.main()
