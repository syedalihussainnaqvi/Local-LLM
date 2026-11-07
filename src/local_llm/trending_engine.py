"""Real-time multi-platform trending topics discovery engine.
Fetches authentic trending topics across 20+ platforms:
  Google Trends, TikTok, Instagram, X (Twitter), YouTube, LinkedIn,
  GitHub, Facebook, Reddit, Hacker News, Pinterest, Twitch, Medium,
  BlueSky, Mastodon, Quora, Steam, Snapchat, Product Hunt, and global news.
"""

import concurrent.futures
import html
import json
import logging
import random
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import urllib.parse

import requests
from . import config

logger = logging.getLogger("trending_engine")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
]


def _random_ua() -> dict:
    return {"User-Agent": random.choice(USER_AGENTS)}


def _clean_text(text: str) -> str:
    """Clean HTML entities and extra whitespace."""
    if not text:
        return ""
    unescaped = html.unescape(text)
    cleaned = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", cleaned).strip()


class TrendingEngine:
    """Aggregates and normalizes real-time trending topics across ALL major platforms."""

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ── 1. Google Trends ────────────────────────────────────────────────────
    def fetch_google_trends(self, geo: str = "US", limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch real-time daily trending topics from Google Trends RSS feed."""
        url = f"https://trends.google.com/trending/rss?geo={geo}"
        trends = []
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                items = root.findall(".//item")
                for item in items[:limit]:
                    title_elem = item.find("title")
                    desc_elem = item.find("description")
                    link_elem = item.find("link")
                    pub_elem = item.find("pubDate")
                    approx_elem = item.find("{https://trends.google.com/trending/rss}approx_traffic")
                    title = _clean_text(title_elem.text if title_elem is not None else "")
                    snippet = _clean_text(desc_elem.text if desc_elem is not None else "")
                    link = link_elem.text if link_elem is not None else ""
                    approx = approx_elem.text if approx_elem is not None else ""
                    pub = pub_elem.text if pub_elem is not None else ""
                    if title:
                        trends.append({
                            "title": title, "platform": "Google Trends", "category": "general",
                            "snippet": snippet or f"High search interest today ({approx} searches)",
                            "url": link or f"https://www.google.com/search?q={urllib.parse.quote_plus(title)}",
                            "metric": approx or "Trending", "published": pub, "icon": "📈",
                        })
        except Exception as e:
            logger.debug(f"Google Trends fetch error: {e}")
        return trends

    # ── 2. Reddit ───────────────────────────────────────────────────────────
    def fetch_reddit_trends(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch hot trending discussions from Reddit r/popular RSS feed."""
        url = "https://www.reddit.com/r/popular/.rss?limit=15"
        trends = []
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall(".//atom:entry", ns) or root.findall(".//entry")
                for entry in entries[:limit]:
                    title_elem = entry.find("atom:title", ns) or entry.find("title")
                    link_elem = entry.find("atom:link", ns) or entry.find("link")
                    author_elem = entry.find(".//atom:name", ns)
                    title = _clean_text(title_elem.text if title_elem is not None else "")
                    link = link_elem.get("href", "") if link_elem is not None else ""
                    author = author_elem.text if author_elem is not None else "Reddit"
                    if title and not title.lower().startswith("[promoted]"):
                        trends.append({
                            "title": title, "platform": "Reddit", "category": "social",
                            "snippet": f"Viral discussion on Reddit ({author})",
                            "url": link, "metric": "Viral Thread", "published": "Recent", "icon": "🔴",
                        })
        except Exception as e:
            logger.debug(f"Reddit trends fetch error: {e}")
        return trends

    # ── 3. GitHub ───────────────────────────────────────────────────────────
    def fetch_github_trends(self, limit: int = 8) -> List[Dict[str, Any]]:
        """Fetch trending open-source repositories from GitHub API."""
        url = "https://api.github.com/search/repositories?q=stars:>1000+pushed:>2024-01-01&sort=updated&order=desc&per_page=12"
        trends = []
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                for item in resp.json().get("items", [])[:limit]:
                    name = item.get("full_name", "")
                    desc = item.get("description", "") or "Trending Open Source Software"
                    stars = item.get("stargazers_count", 0)
                    lang = item.get("language", "Code") or "Code"
                    if name:
                        trends.append({
                            "title": f"{name} ({lang})", "platform": "GitHub", "category": "tech",
                            "snippet": f"{desc} • {stars:,} stars",
                            "url": item.get("html_url", ""), "metric": f"{stars:,} ⭐",
                            "published": "Today", "icon": "🐙",
                        })
        except Exception as e:
            logger.debug(f"GitHub trends fetch error: {e}")
        return trends

    # ── 4. Hacker News ──────────────────────────────────────────────────────
    def fetch_hacker_news_trends(self, limit: int = 8) -> List[Dict[str, Any]]:
        """Fetch top tech stories from Hacker News Firebase API."""
        trends = []
        try:
            resp = self.session.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=self.timeout)
            if resp.status_code == 200:
                story_ids = resp.json()[:limit]
                def _fetch_story(sid):
                    try:
                        r = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json", timeout=2.5)
                        return r.json() if r.status_code == 200 else None
                    except Exception:
                        return None
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
                    for item in concurrent.futures.as_completed([ex.submit(_fetch_story, s) for s in story_ids]):
                        d = item.result()
                        if d and d.get("title"):
                            trends.append({
                                "title": _clean_text(d.get("title", "")),
                                "platform": "Hacker News", "category": "tech",
                                "snippet": f"Tech/AI • {d.get('score',0)} pts • {d.get('descendants',0)} comments",
                                "url": d.get("url") or f"https://news.ycombinator.com/item?id={d.get('id')}",
                                "metric": f"{d.get('score',0)} pts", "published": "Today", "icon": "🔶",
                            })
        except Exception as e:
            logger.debug(f"Hacker News trends fetch error: {e}")
        return trends

    # ── 5. Generic Google News RSS helper ───────────────────────────────────
    def fetch_social_news_trends(self, platform_name: str, query: str, limit: int = 6, icon: str = "🌐") -> List[Dict[str, Any]]:
        """Fetch trending topics for any platform via Google News RSS."""
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        trends = []
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:limit]:
                    title_e = item.find("title")
                    link_e = item.find("link")
                    pub_e = item.find("pubDate")
                    desc_e = item.find("description")
                    raw = _clean_text(title_e.text if title_e is not None else "")
                    clean = raw.rsplit(" - ", 1)[0].strip() if " - " in raw else raw
                    if clean:
                        trends.append({
                            "title": clean, "platform": platform_name, "category": "social",
                            "snippet": _clean_text(desc_e.text if desc_e is not None else "") or f"Trending on {platform_name}",
                            "url": link_e.text if link_e is not None else "",
                            "metric": "Viral Trend",
                            "published": pub_e.text if pub_e is not None else "",
                            "icon": icon,
                        })
        except Exception as e:
            logger.debug(f"Social news trends fetch error for {platform_name}: {e}")
        return trends

    # ── 6. TikTok ───────────────────────────────────────────────────────────
    def fetch_tiktok_trends(self, limit: int = 8) -> List[Dict[str, Any]]:
        """Fetch trending TikTok videos and hashtags."""
        results = self.fetch_social_news_trends("TikTok", "site:tiktok.com viral OR trending OR fyp", limit, "🎵")
        if len(results) < 3:
            results.extend(self.fetch_social_news_trends("TikTok", "TikTok trending viral challenge 2025", limit, "🎵"))
        return results[:limit]

    # ── 7. Instagram ────────────────────────────────────────────────────────
    def fetch_instagram_trends(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Fetch trending Instagram Reels and posts."""
        results = self.fetch_social_news_trends("Instagram", "site:instagram.com trending OR reels OR viral", limit, "📸")
        if len(results) < 3:
            results.extend(self.fetch_social_news_trends("Instagram", "Instagram trending reels viral 2025", limit, "📸"))
        return results[:limit]

    # ── 8. X / Twitter ──────────────────────────────────────────────────────
    def fetch_twitter_trends(self, limit: int = 8) -> List[Dict[str, Any]]:
        """Fetch trending X/Twitter hashtags and viral posts."""
        results = self.fetch_social_news_trends("X (Twitter)", "site:x.com OR site:twitter.com trending OR viral", limit, "𝕏")
        if len(results) < 3:
            results.extend(self.fetch_social_news_trends("X (Twitter)", "Twitter trending hashtag viral 2025", limit, "𝕏"))
        return results[:limit]

    # ── 9. YouTube ──────────────────────────────────────────────────────────
    def fetch_youtube_trends(self, limit: int = 8) -> List[Dict[str, Any]]:
        """Fetch trending YouTube videos and Shorts."""
        results = self.fetch_social_news_trends("YouTube", "site:youtube.com trending OR viral OR shorts", limit, "▶️")
        try:
            resp = self.session.get("https://www.youtube.com/feeds/videos.xml?chart=0&hl=en&gl=US", timeout=self.timeout, headers=_random_ua())
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                ns = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}
                for entry in root.findall(".//atom:entry", ns)[:limit]:
                    t = entry.find("atom:title", ns)
                    l = entry.find("atom:link", ns)
                    title = t.text if t is not None else ""
                    link = l.get("href","") if l is not None else ""
                    if title:
                        results.append({"title": title, "platform": "YouTube", "category": "entertainment",
                            "snippet": "Trending YouTube Video", "url": link, "metric": "Trending", "published": "Today", "icon": "▶️"})
        except Exception as e:
            logger.debug(f"YouTube RSS fetch error: {e}")
        return results[:limit]

    # ── 10. LinkedIn ────────────────────────────────────────────────────────
    def fetch_linkedin_trends(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Fetch trending LinkedIn articles and business discussions."""
        results = self.fetch_social_news_trends("LinkedIn", "site:linkedin.com trending OR viral OR pulse", limit, "💼")
        if len(results) < 3:
            results.extend(self.fetch_social_news_trends("LinkedIn", "LinkedIn trending business AI career 2025", limit, "💼"))
        return results[:limit]

    # ── 11. Facebook ────────────────────────────────────────────────────────
    def fetch_facebook_trends(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Fetch trending Facebook posts and viral content."""
        results = self.fetch_social_news_trends("Facebook", "site:facebook.com trending OR viral", limit, "👥")
        if len(results) < 3:
            results.extend(self.fetch_social_news_trends("Facebook", "Facebook trending viral post 2025", limit, "👥"))
        return results[:limit]

    # ── 12. Pinterest ───────────────────────────────────────────────────────
    def fetch_pinterest_trends(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Fetch trending Pinterest pins and searches."""
        results = self.fetch_social_news_trends("Pinterest", "site:pinterest.com trending viral", limit, "📌")
        try:
            resp = self.session.get("https://trends.pinterest.com/api/trending?limit=10&country=US&category=all", timeout=self.timeout, headers=_random_ua())
            if resp.status_code == 200:
                for trend in resp.json().get("data", {}).get("trending", [])[:limit]:
                    term = trend.get("query", "")
                    if term:
                        results.append({"title": term, "platform": "Pinterest", "category": "lifestyle",
                            "snippet": f"Trending search on Pinterest: '{term}'",
                            "url": f"https://pinterest.com/search/pins/?q={urllib.parse.quote_plus(term)}",
                            "metric": "Trending", "published": "Today", "icon": "📌"})
        except Exception as e:
            logger.debug(f"Pinterest trends API error: {e}")
        return results[:limit]

    # ── 13. Twitch ──────────────────────────────────────────────────────────
    def fetch_twitch_trends(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Fetch trending Twitch streams and games."""
        return self.fetch_social_news_trends("Twitch", "twitch trending live stream game 2025", limit, "🎮")

    # ── 14. Medium ──────────────────────────────────────────────────────────
    def fetch_medium_trends(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Fetch trending Medium articles via RSS feed."""
        trends = []
        try:
            resp = self.session.get("https://medium.com/feed/topic/technology", timeout=self.timeout, headers=_random_ua())
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:limit]:
                    t = item.find("title")
                    l = item.find("link")
                    d = item.find("description")
                    p = item.find("pubDate")
                    title = _clean_text(t.text if t is not None else "")
                    if title:
                        trends.append({"title": title, "platform": "Medium", "category": "tech",
                            "snippet": _clean_text(d.text if d is not None else "")[:200] or "Trending tech article on Medium",
                            "url": l.text if l is not None else "", "metric": "Top Story",
                            "published": p.text if p is not None else "", "icon": "✍️"})
        except Exception as e:
            logger.debug(f"Medium trends fetch error: {e}")
        return trends

    # ── 15. BlueSky ─────────────────────────────────────────────────────────
    def fetch_bluesky_trends(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Fetch trending BlueSky topics via public AT Protocol API."""
        trends = []
        try:
            resp = self.session.get("https://public.api.bsky.app/xrpc/app.bsky.feed.getTrendingTopics?limit=10", timeout=self.timeout, headers=_random_ua())
            if resp.status_code == 200:
                for topic in resp.json().get("topics", [])[:limit]:
                    tag = topic.get("topic", "")
                    cnt = topic.get("postCount", 0)
                    if tag:
                        trends.append({"title": f"#{tag}", "platform": "BlueSky", "category": "social",
                            "snippet": f"Trending on BlueSky with {cnt:,} posts",
                            "url": f"https://bsky.app/search?q={urllib.parse.quote_plus(tag)}",
                            "metric": f"{cnt:,} posts", "published": "Now", "icon": "🦋"})
        except Exception as e:
            logger.debug(f"BlueSky trends fetch error: {e}")
        if not trends:
            trends.extend(self.fetch_social_news_trends("BlueSky", "BlueSky trending viral 2025", limit, "🦋"))
        return trends[:limit]

    # ── 16. Mastodon ────────────────────────────────────────────────────────
    def fetch_mastodon_trends(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Fetch trending hashtags from Mastodon.social public API."""
        trends = []
        try:
            resp = self.session.get("https://mastodon.social/api/v1/trends/tags?limit=10", timeout=self.timeout, headers=_random_ua())
            if resp.status_code == 200:
                for tag in resp.json()[:limit]:
                    name = tag.get("name", "")
                    uses = tag.get("history", [{}])[0].get("uses", "0") if tag.get("history") else "?"
                    if name:
                        trends.append({"title": f"#{name}", "platform": "Mastodon", "category": "social",
                            "snippet": f"Trending on Mastodon with {uses} recent posts",
                            "url": f"https://mastodon.social/tags/{name}",
                            "metric": f"{uses} posts", "published": "Now", "icon": "🐘"})
        except Exception as e:
            logger.debug(f"Mastodon trends fetch error: {e}")
        return trends

    # ── 17. Snapchat ────────────────────────────────────────────────────────
    def fetch_snapchat_trends(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self.fetch_social_news_trends("Snapchat", "snapchat trending discover viral 2025", limit, "👻")

    # ── 18. Quora ───────────────────────────────────────────────────────────
    def fetch_quora_trends(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self.fetch_social_news_trends("Quora", "site:quora.com trending OR popular OR most answered", limit, "❓")

    # ── 19. Steam / Gaming ──────────────────────────────────────────────────
    def fetch_steam_trends(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch top-selling games from Steam Valve API."""
        trends = []
        try:
            resp = self.session.get("https://store.steampowered.com/api/featuredcategories?l=english&cc=US", timeout=self.timeout, headers=_random_ua())
            if resp.status_code == 200:
                for game in resp.json().get("top_sellers", {}).get("items", [])[:limit]:
                    name = game.get("name", "")
                    app_id = game.get("id", "")
                    price = game.get("final_formatted", "Free")
                    disc = game.get("discount_percent", 0)
                    if name:
                        trends.append({"title": name, "platform": "Steam", "category": "gaming",
                            "snippet": f"Top Selling on Steam • Price: {price}" + (f" ({disc}% OFF)" if disc else ""),
                            "url": f"https://store.steampowered.com/app/{app_id}",
                            "metric": "Top Seller", "published": "Today", "icon": "🎮"})
        except Exception as e:
            logger.debug(f"Steam trends fetch error: {e}")
        return trends

    # ── 20. Global Breaking News (BBC, Reuters, CNN, AP, Al Jazeera, NPR) ───
    def fetch_global_news_trends(self, limit: int = 8) -> List[Dict[str, Any]]:
        """Fetch breaking global news from major RSS feeds concurrently."""
        feeds = [
            ("BBC", "http://feeds.bbci.co.uk/news/rss.xml", "📺"),
            ("Reuters", "https://feeds.reuters.com/reuters/topNews", "📰"),
            ("AP News", "https://rsshub.app/apnews/topics/ap-top-news", "🗞️"),
            ("CNN", "http://rss.cnn.com/rss/edition.rss", "📡"),
            ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml", "🌍"),
            ("NPR", "https://feeds.npr.org/1001/rss.xml", "🎙️"),
        ]
        all_items = []

        def _fetch_feed(name, feed_url, icon):
            items = []
            try:
                resp = self.session.get(feed_url, timeout=self.timeout, headers=_random_ua())
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.findall(".//item")[:3]:
                        t = item.find("title")
                        l = item.find("link")
                        d = item.find("description")
                        p = item.find("pubDate")
                        title = _clean_text(t.text if t is not None else "")
                        if title:
                            items.append({"title": title, "platform": name, "category": "news",
                                "snippet": _clean_text(d.text if d is not None else "")[:200] or f"Breaking news from {name}",
                                "url": l.text if l is not None else "", "metric": "Breaking News",
                                "published": p.text if p is not None else "", "icon": icon})
            except Exception as e:
                logger.debug(f"News feed fetch error for {name}: {e}")
            return items

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            for f in concurrent.futures.as_completed([ex.submit(_fetch_feed, n, u, ico) for n, u, ico in feeds]):
                try:
                    all_items.extend(f.result())
                except Exception:
                    pass
        return all_items[:limit]

    # ── 21. Product Hunt ────────────────────────────────────────────────────
    def fetch_producthunt_trends(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch trending products from Product Hunt RSS."""
        trends = []
        try:
            resp = self.session.get("https://www.producthunt.com/feed", timeout=self.timeout, headers=_random_ua())
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:limit]:
                    t = item.find("title")
                    l = item.find("link")
                    d = item.find("description")
                    title = _clean_text(t.text if t is not None else "")
                    if title:
                        trends.append({"title": title, "platform": "Product Hunt", "category": "tech",
                            "snippet": _clean_text(d.text if d is not None else "")[:200] or "Trending new product on Product Hunt",
                            "url": l.text if l is not None else "", "metric": "🚀 Trending",
                            "published": "Today", "icon": "🚀"})
        except Exception as e:
            logger.debug(f"ProductHunt trends fetch error: {e}")
        return trends

    # ── Master aggregator (all platforms, concurrent) ────────────────────────
    def get_all_trending_topics(self, category: str = "all", limit: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch and synthesize trending topics across ALL 20+ platforms concurrently.
        Categories: 'all', 'general', 'tech', 'social', 'entertainment', 'news', 'gaming', 'lifestyle'
        """
        results: List[Dict[str, Any]] = []
        fetch_tasks = [
            ("google_trends", lambda: self.fetch_google_trends("US", 8)),
            ("tiktok",        self.fetch_tiktok_trends),
            ("twitter",       self.fetch_twitter_trends),
            ("youtube",       self.fetch_youtube_trends),
            ("instagram",     self.fetch_instagram_trends),
            ("linkedin",      self.fetch_linkedin_trends),
            ("facebook",      self.fetch_facebook_trends),
            ("github",        self.fetch_github_trends),
            ("reddit",        self.fetch_reddit_trends),
            ("hacker_news",   self.fetch_hacker_news_trends),
            ("pinterest",     self.fetch_pinterest_trends),
            ("twitch",        self.fetch_twitch_trends),
            ("medium",        self.fetch_medium_trends),
            ("bluesky",       self.fetch_bluesky_trends),
            ("mastodon",      self.fetch_mastodon_trends),
            ("snapchat",      self.fetch_snapchat_trends),
            ("quora",         self.fetch_quora_trends),
            ("steam",         self.fetch_steam_trends),
            ("global_news",   self.fetch_global_news_trends),
            ("producthunt",   self.fetch_producthunt_trends),
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            future_map = {executor.submit(fn): name for name, fn in fetch_tasks}
            try:
                for future in concurrent.futures.as_completed(future_map, timeout=self.timeout + 4):
                    src = future_map[future]
                    try:
                        items = future.result()
                        results.extend(items)
                    except Exception as e:
                        logger.debug(f"Source {src} execution failed: {e}")
            except TimeoutError:
                logger.debug("Some trending sources took too long; continuing with completed ones")
            except Exception as e:
                logger.debug(f"Trending gathering exception: {e}")

        junk_phrases = ["log in", "sign up", "sign in", "login", "page not found", "404", "access denied", "cookie policy"]
        seen = set()
        deduped = []
        for item in results:
            t_low = item["title"].lower()
            if any(junk in t_low for junk in junk_phrases):
                continue
            key = re.sub(r"[^a-z0-9]", "", t_low)[:30]
            if key and key not in seen:
                seen.add(key)
                if category == "all" or item.get("category") == category:
                    deduped.append(item)
        return deduped[:limit]

    def build_trending_dossier_text(self, limit: int = 20) -> str:
        """Build a comprehensive textual dossier of trending topics for LLM prompting."""
        topics = self.get_all_trending_topics(category="all", limit=limit)
        if not topics:
            return "No real-time trending topics could be fetched at this moment."
        lines = ["=== CURRENT REAL-TIME TRENDING TOPICS ACROSS THE INTERNET ==="]
        for idx, t in enumerate(topics, 1):
            lines.append(
                f"{idx}. [{t['platform']}] {t['title']} ({t['metric']})\n"
                f"   Summary: {t['snippet']}\n"
                f"   URL: {t['url']}"
            )
        lines.append("=== END OF TRENDING TOPICS ===")
        return "\n".join(lines)

    def fetch_trending_for_script(self, topic_hint: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch the most viral trending topics relevant to a given script topic."""
        all_trends = self.get_all_trending_topics(limit=limit * 3)
        if topic_hint:
            hint_lower = topic_hint.lower()
            scored = sorted(
                [(sum(1 for w in hint_lower.split() if w in t["title"].lower() or w in t["snippet"].lower()), t) for t in all_trends],
                key=lambda x: x[0], reverse=True
            )
            top = [t for s, t in scored if s > 0][:limit]
            if top:
                return top
        return all_trends[:limit]


# ── Singleton ────────────────────────────────────────────────────────────────
_GLOBAL_TRENDING: Optional[TrendingEngine] = None


def get_trending_engine() -> TrendingEngine:
    global _GLOBAL_TRENDING
    if _GLOBAL_TRENDING is None:
        _GLOBAL_TRENDING = TrendingEngine()
    return _GLOBAL_TRENDING
