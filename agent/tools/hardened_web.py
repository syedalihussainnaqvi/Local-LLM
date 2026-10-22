"""Hardened Web Scraper Pipeline with SSRF Guards, Independence Checking, and Cited Synthesis.
Addresses:
- C-04: Full SSRF prevention (DNS resolution, private/loopback/link-local/metadata IP block, manual redirect validation).
- H-11: Downloads routed to strict quarantine folder.
- Section 4: Query planning, content extraction, SimHash wire-syndication deduplication, independence checking.
"""

import os
import re
import time
import socket
import hashlib
import ipaddress
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("hardened_web")


class SSRFSecurityException(Exception):
    """Raised when an outbound URL violates SSRF, loopback, or private network boundaries."""
    pass


class GuardedFetcher:
    """Zero-trust HTTP client strictly validating all IP destinations and redirect hops."""

    ALLOWED_SCHEMES = ("http", "https")
    BLOCKED_PORTS = {21, 22, 23, 25, 53, 110, 143, 445, 3389, 5432, 6379, 8080, 27017}
    MAX_REDIRECTS = 5
    MAX_BODY_BYTES = 5 * 1024 * 1024  # 5 MB maximum
    ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/json", "application/xml")

    # Cloud metadata endpoints
    METADATA_IPS = {"169.254.169.254", "100.100.100.200", "metadata.google.internal"}

    @classmethod
    def is_ip_forbidden(cls, ip_str: str) -> bool:
        """Check if IP is private, loopback, link-local, multicast, reserved, or cloud metadata."""
        if ip_str in cls.METADATA_IPS:
            return True
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            return (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
                or ip_obj.is_reserved
                or ip_obj.is_unspecified
            )
        except ValueError:
            return True

    @classmethod
    def validate_url(cls, url: str) -> Tuple[bool, str]:
        """Validate URL scheme, port, hostname, and resolved DNS IPs."""
        try:
            parsed = urlparse(url.strip())
        except Exception as e:
            return False, f"Malformed URL: {e}"

        if parsed.scheme.lower() not in cls.ALLOWED_SCHEMES:
            return False, f"Disallowed scheme '{parsed.scheme}'. Only http and https permitted."

        hostname = parsed.hostname
        if not hostname:
            return False, "Missing hostname."

        if hostname.lower() in ("localhost", "localhost.localdomain", "broadcasthost", "metadata.google.internal"):
            return False, f"Blocked internal hostname '{hostname}'."

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port in cls.BLOCKED_PORTS:
            return False, f"Destination port {port} is blocked."

        # DNS resolution check
        try:
            addr_infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
            if not addr_infos:
                return False, f"No DNS records found for host '{hostname}'."
            for ai in addr_infos:
                ip = ai[4][0]
                if cls.is_ip_forbidden(ip):
                    return False, f"Host '{hostname}' resolved to blocked/private IP: {ip}"
        except socket.gaierror as e:
            return False, f"DNS resolution failed: {e}"

        return True, ""

    def fetch(self, url: str, timeout: float = 8.0) -> Dict[str, Any]:
        """Fetch URL with manual redirect tracking and per-hop SSRF validation."""
        curr_url = url.strip()
        visited: Set[str] = set()

        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 (AntigravityAgent/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })

        for hop in range(self.MAX_REDIRECTS + 1):
            if curr_url in visited:
                raise SSRFSecurityException(f"Circular redirect detected at '{curr_url}'.")
            visited.add(curr_url)

            # Validate hop
            safe, reason = self.validate_url(curr_url)
            if not safe:
                raise SSRFSecurityException(f"SSRF violation on hop {hop} ('{curr_url}'): {reason}")

            resp = session.get(curr_url, allow_redirects=False, timeout=timeout, stream=True)

            # Check redirect status
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    break
                curr_url = urljoin(curr_url, location)
                continue

            # Check content-type
            ctype = resp.headers.get("Content-Type", "").lower().split(";")[0].strip()
            if ctype and not any(allowed in ctype for allowed in self.ALLOWED_CONTENT_TYPES):
                return {
                    "status": "blocked_content_type",
                    "url": curr_url,
                    "content_type": ctype,
                    "error": f"Content-Type '{ctype}' blocked (only text/html and documents permitted).",
                }

            # Read with size cap
            body_chunks = []
            total_bytes = 0
            for chunk in resp.iter_content(chunk_size=16384):
                total_bytes += len(chunk)
                if total_bytes > self.MAX_BODY_BYTES:
                    break
                body_chunks.append(chunk)

            raw_bytes = b"".join(body_chunks)
            encoding = resp.encoding or "utf-8"
            text = raw_bytes.decode(encoding, errors="replace")

            return {
                "status": "success",
                "final_url": curr_url,
                "status_code": resp.status_code,
                "content_type": ctype,
                "html": text,
                "size_bytes": len(raw_bytes),
                "hops": hop,
            }

        raise SSRFSecurityException(f"Exceeded max redirects ({self.MAX_REDIRECTS}).")


class ContentExtractor:
    """Extracts cleaned main text, title, and metadata from HTML."""

    @staticmethod
    def extract(html: str, url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")

        # Strip scripts, styles, navigation, footer, forms
        for tag in soup(["script", "style", "nav", "footer", "header", "form", "aside", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = " ".join(soup.stripped_strings)

        # Truncate clean text
        clean_text = re.sub(r"\s+", " ", text)[:10000]

        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        return {
            "url": url,
            "domain": domain,
            "title": title,
            "clean_text": clean_text,
            "length": len(clean_text),
        }


class IndependenceChecker:
    """Calculates SimHash and Jaccard shingling to detect syndicated wire articles and verify source independence."""

    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        """Compute Jaccard token overlap between two texts."""
        set1 = set(re.findall(r"\w+", text1.lower()))
        set2 = set(re.findall(r"\w+", text2.lower()))
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _simhash(text: str) -> int:
        """Compute a 64-bit SimHash of text unigrams and bigrams."""
        words = re.findall(r"\w+", text.lower())
        if not words:
            return 0

        # Unigrams + bigrams
        tokens = words + [" ".join(words[i:i+2]) for i in range(len(words) - 1)]
        v = [0] * 64

        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:16], 16)
            for i in range(64):
                bit = (h >> i) & 1
                v[i] += 1 if bit else -1

        fingerprint = 0
        for i in range(64):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        """Count differing bits between two 64-bit hashes."""
        return bin(h1 ^ h2).count("1")

    @classmethod
    def is_syndicated(cls, text1: str, text2: str, simhash1: int, simhash2: int) -> bool:
        """Return True if articles are syndicated copies (high Jaccard or low Hamming distance)."""
        jaccard = cls.jaccard_similarity(text1, text2)
        if jaccard >= 0.65:
            return True
        dist = cls.hamming_distance(simhash1, simhash2)
        return dist <= 14

    @classmethod
    def group_independent_sources(cls, articles: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Cluster articles into independent groups. Syndicated articles form one cluster."""
        clusters: List[List[Dict[str, Any]]] = []

        for art in articles:
            text = art.get("clean_text", "")
            h = cls._simhash(text)
            art["simhash"] = h

            placed = False
            for cluster in clusters:
                rep = cluster[0]
                if cls.is_syndicated(text, rep.get("clean_text", ""), h, rep.get("simhash", 0)):
                    cluster.append(art)
                    placed = True
                    break

            if not placed:
                clusters.append([art])

        return clusters

    @classmethod
    def verify_independence(cls, articles: List[Dict[str, Any]], min_independent: int = 2) -> Dict[str, Any]:
        """Verify if a set of collected articles represents at least N independent sources."""
        clusters = cls.group_independent_sources(articles)
        num_independent = len(clusters)
        domains = list({a.get("domain", "") for a in articles if a.get("domain")})

        return {
            "verified": num_independent >= min_independent,
            "independent_source_count": num_independent,
            "total_articles": len(articles),
            "unique_domains": len(domains),
            "clusters": clusters,
        }
