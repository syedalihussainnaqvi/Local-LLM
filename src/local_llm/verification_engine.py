"""Cross-Platform News Verification Engine.
Verifies breaking news, claims, and viral stories across major news outlets,
social media platforms (X/Twitter, Reddit, YouTube, TikTok), and fact-checking sources.
"""

import concurrent.futures
import html
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse

import requests
from bs4 import BeautifulSoup

from . import config
from .social_scraper import get_social_scraper
from .web_scraper import validate_url_safety

logger = logging.getLogger("verification_engine")

TOP_VERIFIED_NEWS_OUTLETS = [
    "Reuters", "Associated Press", "AP News", "BBC", "BBC News",
    "The Guardian", "The New York Times", "The Washington Post",
    "CNN", "Bloomberg", "Al Jazeera", "Financial Times", "DW News",
    "France 24", "NPR", "CBS News", "NBC News", "ABC News", "Wall Street Journal"
]

FACT_CHECK_DOMAINS = [
    "snopes.com", "factcheck.org", "apnews.com/hub/ap-fact-check",
    "reuters.com/fact-check", "politifact.com", "fullfact.org"
]


class CrossPlatformVerificationEngine:
    """Orchestrates multi-platform truth verification, consensus scoring, and dossier generation."""

    def __init__(self, timeout: float = 4.0):
        self.timeout = timeout
        self.social_scraper = get_social_scraper()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })

    def search_news_coverage(self, claim: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Search top journalistic news outlets via Google News RSS."""
        clean_claim = claim.strip()
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(clean_claim)}&hl=en-US&gl=US&ceid=US:en"
        articles = []

        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                items = root.findall(".//item")
                for item in items[:limit]:
                    title_elem = item.find("title")
                    link_elem = item.find("link")
                    pub_elem = item.find("pubDate")
                    desc_elem = item.find("description")
                    source_elem = item.find("source")

                    if title_elem is None or link_elem is None:
                        continue

                    raw_title = title_elem.text or ""
                    link = link_elem.text or ""
                    pub = pub_elem.text if pub_elem is not None else ""
                    snippet = ""
                    if desc_elem is not None and desc_elem.text:
                        s = BeautifulSoup(desc_elem.text, "html.parser")
                        snippet = s.get_text(separator=" ", strip=True)

                    channel = source_elem.text.strip() if source_elem is not None and source_elem.text else ""
                    if not channel and " - " in raw_title:
                        channel = raw_title.rsplit(" - ", 1)[-1].strip()

                    clean_title = raw_title.rsplit(" - ", 1)[0].strip() if " - " in raw_title else raw_title

                    # Check if channel is recognized top news source
                    is_major = any(major.lower() in channel.lower() for major in TOP_VERIFIED_NEWS_OUTLETS)

                    safe, _ = validate_url_safety(link)
                    if not safe:
                        continue

                    articles.append({
                        "tier": "news",
                        "source": channel or "News Publisher",
                        "title": clean_title,
                        "url": link,
                        "snippet": snippet,
                        "published": pub,
                        "is_major_outlet": is_major,
                    })
        except Exception as e:
            logger.debug(f"News verification search error: {e}")

        return articles

    def search_fact_checkers(self, claim: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search dedicated fact-checking publications."""
        q = f"{claim} (fact check OR snopes OR politifact OR reuters fact check)"
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"
        checks = []

        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                items = root.findall(".//item")
                for item in items[:limit]:
                    title_elem = item.find("title")
                    link_elem = item.find("link")
                    desc_elem = item.find("description")
                    pub_elem = item.find("pubDate")
                    source_elem = item.find("source")

                    if title_elem is None or link_elem is None:
                        continue

                    raw_title = title_elem.text or ""
                    link = link_elem.text or ""
                    pub = pub_elem.text if pub_elem is not None else ""
                    snippet = ""
                    if desc_elem is not None and desc_elem.text:
                        s = BeautifulSoup(desc_elem.text, "html.parser")
                        snippet = s.get_text(separator=" ", strip=True)

                    channel = source_elem.text.strip() if source_elem is not None and source_elem.text else "Fact Checker"
                    clean_title = raw_title.rsplit(" - ", 1)[0].strip() if " - " in raw_title else raw_title

                    safe, _ = validate_url_safety(link)
                    if not safe:
                        continue

                    checks.append({
                        "tier": "fact_check",
                        "source": channel,
                        "title": clean_title,
                        "url": link,
                        "snippet": snippet,
                        "published": pub,
                    })
        except Exception as e:
            logger.debug(f"Fact check search error: {e}")

        return checks

    def verify_claim(self, claim: str) -> Dict[str, Any]:
        """
        Cross-verify a news claim across 3 tiers:
        1. Journalistic news outlets (BBC, Reuters, AP, etc.)
        2. Social media platforms (X, Reddit, YouTube, TikTok)
        3. Dedicated fact-checking organizations
        """
        news_results = []
        fact_results = []
        social_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            fut_news = executor.submit(self.search_news_coverage, claim, 12)
            fut_facts = executor.submit(self.search_fact_checkers, claim, 6)
            fut_social = executor.submit(
                self.social_scraper.search_all_social_media,
                claim,
                ["x", "reddit", "youtube", "tiktok"],
                2
            )

            try:
                news_results = fut_news.result()
            except Exception as e:
                logger.debug(f"Verification news future error: {e}")

            try:
                fact_results = fut_facts.result()
            except Exception as e:
                logger.debug(f"Verification facts future error: {e}")

            try:
                social_data = fut_social.result()
                social_results = social_data.get("results", [])
            except Exception as e:
                logger.debug(f"Verification social future error: {e}")

        # Assess distinct major channels reporting
        major_news_channels = list(dict.fromkeys(
            item["source"] for item in news_results if item.get("is_major_outlet")
        ))
        all_news_channels = list(dict.fromkeys(item["source"] for item in news_results))

        # Check for debunking signals in titles/snippets
        claim_lower = claim.lower()
        debunk_keywords = ["false", "debunked", "hoax", "fake", "misleading", "fabricated", "not true", "unsubstantiated"]
        has_debunk = any(
            any(dk in (r.get("title", "") + " " + r.get("snippet", "")).lower() for dk in debunk_keywords)
            for r in fact_results + news_results
        )

        # Determine consensus verdict
        if has_debunk:
            verdict = "DEBUNKED / FALSE"
            verdict_badge = "❌ DEBUNKED / FALSE"
            confidence = "High"
            explanation = "Fact-checking organizations or major reporting outlets have identified this claim as false, misleading, or a hoax."
        elif len(major_news_channels) >= 2:
            verdict = "VERIFIED / CONFIRMED"
            verdict_badge = "✅ VERIFIED / CONFIRMED"
            confidence = "High"
            explanation = f"Confirmed independently by multiple major news organizations: {', '.join(major_news_channels[:4])}."
        elif len(all_news_channels) >= 2:
            verdict = "REPORTED / CREDIBLE"
            verdict_badge = "ℹ️ REPORTED BY MULTIPLE OUTLETS"
            confidence = "Medium"
            explanation = f"Reported by {len(all_news_channels)} news sources ({', '.join(all_news_channels[:4])}), pending further independent confirmation."
        elif social_results:
            verdict = "UNVERIFIED / VIRAL CLAIM"
            verdict_badge = "⚠️ UNVERIFIED VIRAL CLAIM"
            confidence = "Low"
            explanation = "Circulating on social media (X/Twitter, TikTok, Reddit), but not yet verified or reported by major independent news organizations."
        else:
            verdict = "INSUFFICIENT DATA"
            verdict_badge = "❓ INSUFFICIENT DATA"
            confidence = "Low"
            explanation = "No authoritative confirmation or widespread reports found across news or social media platforms."

        # Compile candidates for RAG grounding
        rag_candidates = []
        for n in news_results[:6]:
            rag_candidates.append({
                "url": n["url"],
                "title": f"[{n['source']}] {n['title']}",
                "snippet": n["snippet"],
                "source": n["source"],
                "channel": n["source"],
                "category": "news_verification",
                "is_verified_channel": n.get("is_major_outlet", False),
            })

        for fc in fact_results[:3]:
            rag_candidates.append({
                "url": fc["url"],
                "title": f"[Fact Check: {fc['source']}] {fc['title']}",
                "snippet": fc["snippet"],
                "source": fc["source"],
                "channel": fc["source"],
                "category": "fact_check",
            })

        for soc in social_results[:4]:
            rag_candidates.append({
                "url": soc["url"],
                "title": f"[{soc['platform']}] {soc['title']}",
                "snippet": soc["snippet"],
                "source": soc["platform"],
                "channel": soc["platform"],
                "category": "social_reaction",
            })

        return {
            "claim": claim,
            "verdict": verdict,
            "verdict_badge": verdict_badge,
            "confidence": confidence,
            "explanation": explanation,
            "major_news_channels": major_news_channels,
            "news_count": len(news_results),
            "social_count": len(social_results),
            "fact_count": len(fact_results),
            "news_evidence": news_results[:6],
            "social_evidence": social_results[:4],
            "fact_evidence": fact_results[:3],
            "rag_candidates": rag_candidates,
        }

    def build_verification_dossier_text(self, claim: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Build formatted verification report suitable for LLM injection."""
        data = self.verify_claim(claim)
        today_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")

        lines = [
            f"=== MULTI-PLATFORM NEWS & FACT VERIFICATION DOSSIER ===",
            f"Target Claim / News Query: \"{claim}\"",
            f"Date of Verification: {today_str}",
            f"Consensus Verdict: {data['verdict_badge']} (Confidence: {data['confidence']})",
            f"Summary: {data['explanation']}\n",
            f"1. Verified News Coverage ({len(data['major_news_channels'])} major outlets confirmed):"
        ]

        if data["news_evidence"]:
            for item in data["news_evidence"][:5]:
                lines.append(f"- [{item['source']}] {item['title']}\n  Details: {item['snippet']}\n  Link: {item['url']}")
        else:
            lines.append("- No major news coverage found.")

        lines.append(f"\n2. Social Media Discussions & Viral Reaction (X, Reddit, TikTok, YouTube):")
        if data["social_evidence"]:
            for item in data["social_evidence"][:4]:
                lines.append(f"- [{item['platform']}] {item['title']}\n  Discussion: {item['snippet']}\n  Link: {item['url']}")
        else:
            lines.append("- No significant social media discussion detected.")

        if data["fact_evidence"]:
            lines.append(f"\n3. Official Fact Checks & Rebuttals:")
            for item in data["fact_evidence"][:3]:
                lines.append(f"- [{item['source']}] {item['title']}\n  Findings: {item['snippet']}\n  Link: {item['url']}")

        lines.append("=== END OF VERIFICATION DOSSIER ===")
        return "\n".join(lines), data["rag_candidates"]


# Singleton
_GLOBAL_VERIFIER: Optional[CrossPlatformVerificationEngine] = None

def get_verification_engine() -> CrossPlatformVerificationEngine:
    global _GLOBAL_VERIFIER
    if _GLOBAL_VERIFIER is None:
        _GLOBAL_VERIFIER = CrossPlatformVerificationEngine()
    return _GLOBAL_VERIFIER
