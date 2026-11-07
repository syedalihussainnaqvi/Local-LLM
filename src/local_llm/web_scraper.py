"""Safe, powerful, modular web scraper with SSRF protection, content normalization,
JSON-LD schema parsing, OpenGraph metadata extraction, and multi-platform social media handlers.
"""

import hashlib
import ipaddress
import json
import logging
import random
import re
import socket
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import urllib.robotparser

import requests
from bs4 import BeautifulSoup

from . import config

logger = logging.getLogger("web_scraper")
if not logger.handlers:
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

# Cache robots parsers by scheme+netloc: {netloc: (timestamp, RobotFileParser)}
_ROBOTS_CACHE: Dict[str, Tuple[float, urllib.robotparser.RobotFileParser]] = {}
ROBOTS_CACHE_TTL = 3600  # 1 hour

# Rotating realistic user agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
]


class SecurityException(Exception):
    """Raised when a URL violates security or SSRF safety checks."""
    pass


def is_ip_blocked(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP is private, loopback, link-local, multicast, or reserved."""
    if (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    ):
        return True

    # Cloud metadata service endpoints
    ip_str = str(ip_obj)
    if ip_str in ("169.254.169.254", "100.100.100.200"):
        return True

    return False


def validate_url_safety(url: str) -> Tuple[bool, str]:
    """
    Validate that a URL is safe to fetch:
    - Must have http or https scheme
    - Hostname must resolve to valid public IP addresses
    - Protects against SSRF, internal services, and cloud metadata access
    """
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty."

    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Malformed URL: {e}"

    if parsed.scheme.lower() not in config.ALLOWED_SCHEMES:
        return False, f"Unsupported scheme '{parsed.scheme}'. Only http and https are allowed."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL is missing a valid hostname."

    # Reject localhost name variations
    if hostname.lower() in ("localhost", "localhost.localdomain", "broadcasthost", "metadata.google.internal"):
        return False, f"Access to internal hostname '{hostname}' is blocked."

    # Check port if explicitly specified
    if parsed.port and parsed.port in config.BLOCKED_PORTS:
        return False, f"Port {parsed.port} is blocked for safety."

    # Resolve hostname via DNS and inspect resolved IP addresses
    try:
        addr_infos = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
        if not addr_infos:
            return False, f"DNS resolution failed for hostname '{hostname}'."

        for addr_info in addr_infos:
            sockaddr = addr_info[4]
            ip_str = sockaddr[0]
            try:
                ip_obj = ipaddress.ip_address(ip_str)
                if is_ip_blocked(ip_obj):
                    return False, f"Resolved IP {ip_str} is in a blocked/private range (SSRF protection)."
            except ValueError:
                return False, f"Invalid resolved IP '{ip_str}'."

    except socket.gaierror as e:
        return False, f"DNS resolution failed for '{hostname}': {e}"
    except Exception as e:
        return False, f"Error validating host '{hostname}': {e}"

    return True, ""


def check_robots_txt(url: str, user_agent: str = config.USER_AGENT) -> bool:
    """Check robots.txt for the given URL."""
    try:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        now = time.time()

        if base in _ROBOTS_CACHE:
            ts, parser = _ROBOTS_CACHE[base]
            if now - ts < ROBOTS_CACHE_TTL:
                return parser.can_fetch(user_agent, url)

        robots_url = f"{base}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)

        safe, _ = validate_url_safety(robots_url)
        if safe:
            try:
                resp = requests.get(robots_url, timeout=3.0, headers={"User-Agent": user_agent})
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                else:
                    rp.allow_all = True
            except Exception:
                rp.allow_all = True
        else:
            rp.allow_all = True

        _ROBOTS_CACHE[base] = (now, rp)
        return rp.can_fetch(user_agent, url)
    except Exception as e:
        logger.debug(f"robots.txt check skipped: {e}")
        return True


class SafeWebScraper:
    """Powerful, robust web scraper with rotating browser headers, JSON-LD schema parsing, and social media handlers."""

    def __init__(
        self,
        timeout: float = config.REQUEST_TIMEOUT,
        max_size: int = config.MAX_RESPONSE_SIZE,
        max_redirects: int = config.MAX_REDIRECTS,
        user_agent: Optional[str] = None,
        respect_robots: bool = False,
    ):
        self.timeout = timeout
        self.max_size = max_size
        self.max_redirects = max_redirects
        self.user_agent = user_agent or random.choice(USER_AGENTS)
        self.respect_robots = respect_robots

    def fetch_url(self, url: str) -> Dict[str, Any]:
        """Safely fetch a URL following redirects manually while re-validating each hop."""
        current_url = url.strip()
        redirect_count = 0
        visited_urls = set()

        session = requests.Session()
        ua = random.choice(USER_AGENTS)
        session.headers.update({
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Upgrade-Insecure-Requests": "1",
        })

        while True:
            if current_url in visited_urls:
                raise SecurityException(f"Circular redirect detected at '{current_url}'.")
            visited_urls.add(current_url)

            # Re-validate every hop against SSRF
            is_safe, reason = validate_url_safety(current_url)
            if not is_safe:
                raise SecurityException(f"Target '{current_url}' rejected: {reason}")

            if self.respect_robots and not check_robots_txt(current_url, ua):
                raise PermissionError(f"Scraping disallowed by robots.txt for '{current_url}'.")

            try:
                resp = session.get(
                    current_url,
                    timeout=self.timeout,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.Timeout:
                raise TimeoutError(f"Connection timed out after {self.timeout}s fetching '{current_url}'.")
            except requests.RequestException as e:
                raise ConnectionError(f"Network error requesting '{current_url}': {e}")

            # Check redirects
            if resp.status_code in (301, 302, 303, 307, 308):
                redirect_count += 1
                if redirect_count > self.max_redirects:
                    raise SecurityException(f"Exceeded maximum redirect limit of {self.max_redirects}.")

                location = resp.headers.get("Location")
                if not location:
                    raise ValueError(f"HTTP {resp.status_code} redirect without Location header.")

                current_url = urljoin(current_url, location)
                continue

            # Verify content type
            content_type = resp.headers.get("Content-Type", "").lower()
            allowed_types = ("text/html", "application/xhtml+xml", "text/plain", "application/json", "application/xml")
            if not any(t in content_type for t in allowed_types):
                raise ValueError(
                    f"Unsupported content type '{content_type}'. Only HTML, plain text, and JSON are supported."
                )

            # Stream download to enforce response size ceiling
            chunks = []
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    downloaded += len(chunk)
                    if downloaded > self.max_size:
                        raise ValueError(f"Response size exceeded limit of {self.max_size} bytes.")
                    chunks.append(chunk)

            raw_bytes = b"".join(chunks)
            encoding = resp.encoding or resp.apparent_encoding or "utf-8"
            try:
                html_text = raw_bytes.decode(encoding, errors="replace")
            except Exception:
                html_text = raw_bytes.decode("utf-8", errors="replace")

            return {
                "final_url": current_url,
                "status_code": resp.status_code,
                "html": html_text,
                "headers": dict(resp.headers),
                "content_type": content_type,
            }

    def extract_content(self, html_text: str, source_url: str) -> Dict[str, Any]:
        """Extract title, canonical URL, JSON-LD schema, and clean structured text from HTML."""
        soup = BeautifulSoup(html_text, "html.parser")

        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        elif soup.find("meta", property="og:title"):
            title = soup.find("meta", property="og:title").get("content", "").strip()

        if not title:
            title = urlparse(source_url).path.split("/")[-1] or source_url

        # Extract canonical URL
        canonical_url = source_url
        canonical_tag = soup.find("link", rel="canonical")
        if canonical_tag and canonical_tag.get("href"):
            canonical_url = urljoin(source_url, canonical_tag.get("href").strip())

        # Extract JSON-LD structured schemas (Article, NewsArticle, VideoObject, etc.)
        schema_texts = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                if isinstance(data, list):
                    items = data
                else:
                    items = [data]

                for item in items:
                    if not isinstance(item, dict):
                        continue
                    itype = str(item.get("@type", ""))
                    headline = item.get("headline") or item.get("name")
                    desc = item.get("description") or item.get("articleBody")
                    if headline:
                        schema_texts.append(f"[{itype}] {headline}")
                    if desc and isinstance(desc, str):
                        schema_texts.append(desc[:1500])
            except Exception:
                pass

        # Strip unneeded noise elements
        for element in soup([
            "script", "style", "noscript", "svg", "canvas", "iframe",
            "nav", "footer", "header", "aside", "form", "button", "input"
        ]):
            element.decompose()

        # Build clean textual representation preserving headings & paragraphs
        body = soup.find("body") or soup
        lines = []

        if schema_texts:
            lines.append("## Structured Article Data\n" + "\n".join(schema_texts) + "\n")

        for elem in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "td"]):
            text = elem.get_text(separator=" ", strip=True)
            if not text:
                continue

            tag_name = elem.name.lower()
            if tag_name in ("h1", "h2", "h3", "h4"):
                lines.append(f"\n## {text}\n")
            elif tag_name == "li":
                lines.append(f"- {text}")
            else:
                lines.append(text)

        raw_text = "\n".join(lines) if lines else body.get_text(separator="\n", strip=True)

        # Normalize whitespace
        cleaned = re.sub(r"[ \t]+", " ", raw_text)
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned).strip()

        # Enforce max extracted length
        if len(cleaned) > config.MAX_EXTRACTED_LENGTH:
            cleaned = cleaned[: config.MAX_EXTRACTED_LENGTH]

        if len(cleaned) < 25:
            raise ValueError(f"Extracted content from '{source_url}' contains insufficient usable text.")

        content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        retrieved_at = datetime.now(timezone.utc).isoformat()

        return {
            "url": source_url,
            "canonical_url": canonical_url,
            "title": title,
            "text": cleaned,
            "content_hash": content_hash,
            "retrieved_at": retrieved_at,
            "char_count": len(cleaned),
        }

    def scrape(self, url: str) -> Dict[str, Any]:
        """
        High-level safe scrape method.
        Directs social media domains to specialized extractors, otherwise performs universal web extraction.
        """
        u_lower = url.lower()
        social_domains = ["reddit.com", "github.com", "tiktok.com", "youtube.com", "youtu.be", "instagram.com", "twitter.com", "x.com", "linkedin.com"]

        if any(sd in u_lower for sd in social_domains):
            try:
                from .social_scraper import get_social_scraper
                soc = get_social_scraper().scrape_social_url(url)
                if soc.get("status") == "success":
                    content_hash = hashlib.sha256(soc["text"].encode("utf-8")).hexdigest()
                    return {
                        "status": "success",
                        "error": None,
                        "http_status": 200,
                        "url": url,
                        "canonical_url": url,
                        "title": soc.get("title", url),
                        "text": soc["text"],
                        "content_hash": content_hash,
                        "retrieved_at": soc.get("retrieved_at", datetime.now(timezone.utc).isoformat()),
                        "char_count": soc.get("char_count", len(soc["text"])),
                    }
            except Exception as e:
                logger.debug(f"Specialized social extractor fallback for {url}: {e}")

        # Universal web extraction
        try:
            fetch_result = self.fetch_url(url)
            if fetch_result["status_code"] >= 400:
                return {
                    "url": url,
                    "status": "error",
                    "error": f"HTTP status {fetch_result['status_code']} returned.",
                    "http_status": fetch_result["status_code"],
                }

            extracted = self.extract_content(fetch_result["html"], fetch_result["final_url"])
            return {
                "status": "success",
                "error": None,
                "http_status": fetch_result["status_code"],
                **extracted,
            }
        except (SecurityException, PermissionError, TimeoutError, ConnectionError, ValueError) as err:
            logger.warning(f"Scraping failed for '{url}': {err}")
            return {
                "url": url,
                "status": "error",
                "error": str(err),
                "http_status": 0,
            }
        except Exception as err:
            logger.error(f"Unexpected error scraping '{url}': {err}", exc_info=True)
            return {
                "url": url,
                "status": "error",
                "error": f"Internal scraper error: {err}",
                "http_status": 0,
            }


# Module-level convenience function
def scrape_url(url: str) -> Dict[str, Any]:
    scraper = SafeWebScraper()
    return scraper.scrape(url)
