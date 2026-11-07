"""Tests for SafeWebScraper and SSRF safety validation."""

import unittest
from unittest.mock import MagicMock, patch

from local_llm.web_scraper import SafeWebScraper, SecurityException, is_ip_blocked, validate_url_safety
import ipaddress


class TestWebScraper(unittest.TestCase):

    def setUp(self):
        self.scraper = SafeWebScraper(timeout=2.0, max_size=1024 * 1024)

    def test_ssrf_ip_blocking(self):
        # Loopback
        self.assertTrue(is_ip_blocked(ipaddress.ip_address("127.0.0.1")))
        self.assertTrue(is_ip_blocked(ipaddress.ip_address("::1")))
        # Private ranges
        self.assertTrue(is_ip_blocked(ipaddress.ip_address("10.0.0.1")))
        self.assertTrue(is_ip_blocked(ipaddress.ip_address("192.168.1.1")))
        self.assertTrue(is_ip_blocked(ipaddress.ip_address("172.16.0.5")))
        # Cloud metadata
        self.assertTrue(is_ip_blocked(ipaddress.ip_address("169.254.169.254")))
        # Public IP
        self.assertFalse(is_ip_blocked(ipaddress.ip_address("8.8.8.8")))
        self.assertFalse(is_ip_blocked(ipaddress.ip_address("1.1.1.1")))

    def test_validate_url_safety_schemes(self):
        safe, _ = validate_url_safety("ftp://example.com/file.txt")
        self.assertFalse(safe)
        safe, _ = validate_url_safety("file:///etc/passwd")
        self.assertFalse(safe)
        safe, _ = validate_url_safety("javascript:alert(1)")
        self.assertFalse(safe)
        safe, _ = validate_url_safety("")
        self.assertFalse(safe)

    def test_validate_url_safety_internal_hosts(self):
        safe, reason = validate_url_safety("http://localhost:5000")
        self.assertFalse(safe)
        self.assertIn("blocked", reason.lower())

        safe, reason = validate_url_safety("http://metadata.google.internal/computeMetadata/v1/")
        self.assertFalse(safe)

    def test_html_content_extraction(self):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quantum Computing Overview</title>
            <link rel="canonical" href="https://example.com/quantum">
            <script>alert('malicious script');</script>
            <style>body { color: red; }</style>
        </head>
        <body>
            <nav><a href="/home">Home</a><a href="/about">About</a></nav>
            <h1>Introduction to Quantum Computing</h1>
            <p>Quantum computers utilize qubits that can exist in superpositions of states.</p>
            <h2>Applications</h2>
            <ul>
                <li>Cryptography and prime factorization</li>
                <li>Drug discovery simulation</li>
            </ul>
            <footer>Copyright 2026 Example Corp. All rights reserved.</footer>
        </body>
        </html>
        """
        extracted = self.scraper.extract_content(html, "https://example.com/quantum")
        self.assertEqual(extracted["title"], "Quantum Computing Overview")
        self.assertEqual(extracted["canonical_url"], "https://example.com/quantum")
        self.assertIn("Quantum computers utilize qubits", extracted["text"])
        self.assertIn("Cryptography and prime factorization", extracted["text"])
        # Scripts, styles, nav, footer should be stripped
        self.assertNotIn("malicious script", extracted["text"])
        self.assertNotIn("color: red", extracted["text"])
        self.assertNotIn("Copyright 2026", extracted["text"])
        self.assertIn("content_hash", extracted)
        self.assertGreater(extracted["char_count"], 50)

    def test_empty_page_rejection(self):
        html = "<html><head><title>Empty</title></head><body><p>   </p></body></html>"
        with self.assertRaises(ValueError):
            self.scraper.extract_content(html, "https://example.com/empty")

    @patch("local_llm.web_scraper.validate_url_safety")
    def test_unsafe_redirect_blocking(self, mock_safety):
        # Mock first request returns redirect to private IP
        mock_safety.side_effect = [
            (True, ""),   # first URL is fine
            (False, "Resolved IP 10.0.0.1 is in a blocked/private range"), # redirect target is blocked
        ]

        with patch("requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 302
            mock_resp.headers = {"Location": "http://10.0.0.1/admin"}
            mock_get.return_value = mock_resp

            with self.assertRaises(SecurityException) as ctx:
                self.scraper.fetch_url("http://example.com/redirect-test")
            self.assertIn("rejected", str(ctx.exception).lower())

    def test_oversized_response_handling(self):
        scraper_small = SafeWebScraper(max_size=50)
        with patch("local_llm.web_scraper.validate_url_safety", return_value=(True, "")):
            with patch("requests.Session.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.headers = {"Content-Type": "text/html"}
                mock_resp.iter_content.return_value = [b"A" * 100]
                mock_get.return_value = mock_resp

                with self.assertRaises(ValueError) as ctx:
                    scraper_small.fetch_url("http://example.com/huge")
                self.assertIn("exceeded limit", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
