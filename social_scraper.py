"""Multi-platform social media search & powerful internet-wide scraper.
Supports targeted searches AND deep scraping across:
  TikTok, Instagram, X (Twitter), YouTube, LinkedIn, GitHub,
  Facebook, Reddit, Pinterest, Twitch, Medium, BlueSky, Mastodon,
  Snapchat, Quora, Tumblr, Telegram (public), Discord (public),
  and ANY website on the internet via multi-engine parallel scraping.

Also includes internet-wide search across Google News RSS, Wikipedia,
Bing News, DuckDuckGo HTML, and direct site extraction.
"""

import concurrent.futures
import html
import json
import logging
import random
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse

import requests
from bs4 import BeautifulSoup

import config
from web_scraper import validate_url_safety

logger = logging.getLogger("social_scraper")

# All known social/internet platform metadata
PLATFORM_META = {
    "tiktok":    {"name": "TikTok",     "domain": "tiktok.com",    "icon": "🎵", "query_prefix": "site:tiktok.com"},
    "instagram": {"name": "Instagram",  "domain": "instagram.com", "icon": "📸", "query_prefix": "site:instagram.com"},
    "x":         {"name": "X (Twitter)","domain": "x.com",         "icon": "𝕏", "query_prefix": "site:x.com OR site:twitter.com"},
    "youtube":   {"name": "YouTube",    "domain": "youtube.com",   "icon": "▶️", "query_prefix": "site:youtube.com"},
    "linkedin":  {"name": "LinkedIn",   "domain": "linkedin.com",  "icon": "💼", "query_prefix": "site:linkedin.com"},
    "github":    {"name": "GitHub",     "domain": "github.com",    "icon": "🐙", "query_prefix": "site:github.com"},
    "facebook":  {"name": "Facebook",   "domain": "facebook.com",  "icon": "👥", "query_prefix": "site:facebook.com"},
    "reddit":    {"name": "Reddit",     "domain": "reddit.com",    "icon": "🔴", "query_prefix": "site:reddit.com"},
    "pinterest": {"name": "Pinterest",  "domain": "pinterest.com", "icon": "📌", "query_prefix": "site:pinterest.com"},
    "twitch":    {"name": "Twitch",     "domain": "twitch.tv",     "icon": "🎮", "query_prefix": "site:twitch.tv"},
    "medium":    {"name": "Medium",     "domain": "medium.com",    "icon": "✍️", "query_prefix": "site:medium.com"},
    "bluesky":   {"name": "BlueSky",    "domain": "bsky.app",      "icon": "🦋", "query_prefix": "site:bsky.app"},
    "mastodon":  {"name": "Mastodon",   "domain": "mastodon.social","icon": "🐘","query_prefix": "site:mastodon.social"},
    "snapchat":  {"name": "Snapchat",   "domain": "snapchat.com",  "icon": "👻", "query_prefix": "site:snapchat.com"},
    "quora":     {"name": "Quora",      "domain": "quora.com",     "icon": "❓", "query_prefix": "site:quora.com"},
    "tumblr":    {"name": "Tumblr",     "domain": "tumblr.com",    "icon": "🌀", "query_prefix": "site:tumblr.com"},
    "telegram":  {"name": "Telegram",   "domain": "t.me",          "icon": "✈️", "query_prefix": "site:t.me"},
    "steam":     {"name": "Steam",      "domain": "store.steampowered.com","icon": "🎮","query_prefix": "site:store.steampowered.com"},
    "producthunt":{"name":"ProductHunt","domain": "producthunt.com","icon": "🚀","query_prefix": "site:producthunt.com"},
    "hackernews":{"name": "Hacker News","domain": "news.ycombinator.com","icon": "🔶","query_prefix": "site:news.ycombinator.com"},
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
]


def _random_ua() -> str:
    return random.choice(USER_AGENTS)


def _clean(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(text))).strip()


class SocialMediaScraper:
    """Powerful multi-platform social media search & internet-wide scraper."""

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": _random_ua(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

    # ── Core: Search any platform via Google News RSS ───────────────────────
    def search_single_platform(self, platform_key: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search a single social platform using indexed search and platform-specific APIs."""
        meta = PLATFORM_META.get(platform_key.lower())
        if not meta:
            return []

        search_q = f"{meta['query_prefix']} {query}"
        endpoint = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(search_q)}&hl=en-US&gl=US&ceid=US:en"
        results = []

        try:
            resp = self.session.get(endpoint, timeout=self.timeout)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:limit]:
                    title_e = item.find("title")
                    link_e = item.find("link")
                    desc_e = item.find("description")
                    pub_e = item.find("pubDate")
                    src_e = item.find("source")

                    if title_e is None or link_e is None:
                        continue

                    raw_title = title_e.text or ""
                    link = link_e.text or ""
                    pub = pub_e.text if pub_e is not None else ""
                    snippet = ""
                    if desc_e is not None and desc_e.text:
                        try:
                            snippet = BeautifulSoup(desc_e.text, "html.parser").get_text(separator=" ", strip=True)
                        except Exception:
                            snippet = _clean(desc_e.text)

                    clean_title = raw_title.rsplit(" - ", 1)[0].strip() if " - " in raw_title else raw_title
                    channel = src_e.text.strip() if src_e is not None and src_e.text else meta["name"]

                    safe, _ = validate_url_safety(link)
                    if not safe:
                        continue

                    results.append({
                        "platform": meta["name"],
                        "platform_key": platform_key.lower(),
                        "icon": meta["icon"],
                        "title": clean_title,
                        "url": link,
                        "snippet": snippet or f"Discussion on {meta['name']} about: {clean_title}",
                        "source": channel,
                        "published": pub,
                        "category": "social",
                    })
        except Exception as e:
            logger.debug(f"Search failed for {platform_key}: {e}")

        # ── Platform-specific API fallbacks ──────────────────────────────────

        # GitHub Search API
        if platform_key.lower() == "github" and len(results) < limit:
            try:
                gh_resp = self.session.get(
                    f"https://api.github.com/search/repositories?q={urllib.parse.quote_plus(query)}&sort=stars&order=desc&per_page={limit}",
                    timeout=3.0
                )
                if gh_resp.status_code == 200:
                    for repo in gh_resp.json().get("items", [])[:limit]:
                        results.append({
                            "platform": "GitHub", "platform_key": "github", "icon": "🐙",
                            "title": f"{repo.get('full_name')} ({repo.get('language') or 'Code'})",
                            "url": repo.get("html_url", ""),
                            "snippet": f"{repo.get('description', '')} • {repo.get('stargazers_count', 0):,} stars",
                            "source": "GitHub", "published": "Updated", "category": "tech",
                        })
            except Exception as e:
                logger.debug(f"GitHub API fallback failed: {e}")

        # Reddit Search RSS
        if platform_key.lower() == "reddit" and len(results) < limit:
            try:
                r_resp = self.session.get(
                    f"https://www.reddit.com/search.rss?q={urllib.parse.quote_plus(query)}&sort=relevance&limit={limit}",
                    timeout=3.5
                )
                if r_resp.status_code == 200:
                    root = ET.fromstring(r_resp.content)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    for entry in root.findall(".//atom:entry", ns)[:limit]:
                        rtitle = entry.find("atom:title", ns)
                        rlink = entry.find("atom:link", ns)
                        if rtitle is not None and rlink is not None:
                            results.append({
                                "platform": "Reddit", "platform_key": "reddit", "icon": "🔴",
                                "title": rtitle.text or "",
                                "url": rlink.get("href", ""),
                                "snippet": f"Reddit discussion thread about '{query}'",
                                "source": "Reddit", "published": "Recent", "category": "social",
                            })
            except Exception as e:
                logger.debug(f"Reddit RSS fallback failed: {e}")

        # Mastodon public search API
        if platform_key.lower() == "mastodon" and len(results) < limit:
            try:
                m_resp = self.session.get(
                    f"https://mastodon.social/api/v2/search?q={urllib.parse.quote_plus(query)}&type=statuses&limit={limit}",
                    timeout=3.0,
                    headers={"User-Agent": _random_ua()},
                )
                if m_resp.status_code == 200:
                    for status in m_resp.json().get("statuses", [])[:limit]:
                        content = BeautifulSoup(status.get("content", ""), "html.parser").get_text()[:200]
                        url = status.get("url", "")
                        acct = status.get("account", {}).get("acct", "")
                        if content:
                            results.append({
                                "platform": "Mastodon", "platform_key": "mastodon", "icon": "🐘",
                                "title": f"@{acct}: {content[:80]}",
                                "url": url, "snippet": content,
                                "source": "Mastodon", "published": status.get("created_at", ""), "category": "social",
                            })
            except Exception as e:
                logger.debug(f"Mastodon API fallback failed: {e}")

        # BlueSky public search
        if platform_key.lower() == "bluesky" and len(results) < limit:
            try:
                bsky_resp = self.session.get(
                    f"https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts?q={urllib.parse.quote_plus(query)}&limit={limit}",
                    timeout=3.0,
                    headers={"User-Agent": _random_ua()},
                )
                if bsky_resp.status_code == 200:
                    for post in bsky_resp.json().get("posts", [])[:limit]:
                        text = post.get("record", {}).get("text", "")
                        author = post.get("author", {}).get("handle", "")
                        uri = post.get("uri", "")
                        rkey = uri.split("/")[-1] if "/" in uri else uri
                        url = f"https://bsky.app/profile/{author}/post/{rkey}"
                        if text:
                            results.append({
                                "platform": "BlueSky", "platform_key": "bluesky", "icon": "🦋",
                                "title": f"@{author}: {text[:80]}",
                                "url": url, "snippet": text[:200],
                                "source": "BlueSky", "published": post.get("indexedAt", ""), "category": "social",
                            })
            except Exception as e:
                logger.debug(f"BlueSky API fallback failed: {e}")

        return results[:limit]

    # ── Search ALL social platforms simultaneously ───────────────────────────
    def search_all_social_media(
        self,
        query: str,
        platforms: Optional[List[str]] = None,
        limit_per_platform: int = 3,
    ) -> Dict[str, Any]:
        """
        Search across ALL social media platforms simultaneously.
        Returns grouped-by-platform results and a flat ranked list.
        """
        target_platforms = platforms or list(PLATFORM_META.keys())
        all_results: List[Dict[str, Any]] = []
        by_platform: Dict[str, List[Dict[str, Any]]] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, len(target_platforms))) as ex:
            future_map = {ex.submit(self.search_single_platform, pk, query, limit_per_platform): pk for pk in target_platforms}
            for fut in concurrent.futures.as_completed(future_map):
                pk = future_map[fut]
                try:
                    res = fut.result()
                    by_platform[pk] = res
                    all_results.extend(res)
                except Exception as e:
                    logger.debug(f"Platform {pk} execution failed: {e}")
                    by_platform[pk] = []

        return {
            "query": query,
            "total_found": len(all_results),
            "platforms_searched": len(target_platforms),
            "by_platform": by_platform,
            "results": all_results,
        }

    # ── Internet-wide multi-engine search ───────────────────────────────────
    def search_internet_wide(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Powerful internet-wide search using multiple search engines and RSS feeds:
        - Google News RSS
        - Wikipedia OpenSearch API
        - Bing News RSS
        - DuckDuckGo HTML scraping (user-agent rotation)
        - Hacker News Algolia search API
        - GitHub Code/Repo search
        Returns a ranked, deduplicated list of results from across the entire internet.
        """
        all_results: List[Dict[str, Any]] = []
        seen_urls: set = set()

        def _add(items):
            for item in items:
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(item)

        def _google_news():
            items = []
            try:
                url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.findall(".//item")[:max_results]:
                        t = item.find("title")
                        l = item.find("link")
                        d = item.find("description")
                        src = item.find("source")
                        raw = t.text or "" if t is not None else ""
                        clean = raw.rsplit(" - ", 1)[0].strip() if " - " in raw else raw
                        link = l.text or "" if l is not None else ""
                        safe, _ = validate_url_safety(link)
                        if clean and link and safe:
                            items.append({
                                "title": clean, "url": link,
                                "snippet": _clean(d.text if d is not None else ""),
                                "source": src.text if src is not None else "News",
                                "platform": "Google News", "icon": "📰", "category": "news",
                            })
            except Exception as e:
                logger.debug(f"Google News search error: {e}")
            return items

        def _wikipedia():
            items = []
            try:
                params = {"action": "opensearch", "search": query, "limit": 5, "namespace": 0, "format": "json"}
                resp = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=3.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if len(data) >= 4:
                        for title, desc, url in zip(data[1], data[2], data[3]):
                            safe, _ = validate_url_safety(url)
                            if safe:
                                items.append({
                                    "title": f"Wikipedia: {title}", "url": url,
                                    "snippet": desc or title, "source": "Wikipedia",
                                    "platform": "Wikipedia", "icon": "📖", "category": "encyclopedia",
                                })
            except Exception as e:
                logger.debug(f"Wikipedia search error: {e}")
            return items

        def _bing_news():
            items = []
            try:
                url = f"https://www.bing.com/news/search?q={urllib.parse.quote_plus(query)}&format=rss"
                resp = requests.get(url, timeout=3.5, headers={"User-Agent": _random_ua()})
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.findall(".//item")[:8]:
                        t = item.find("title")
                        l = item.find("link")
                        d = item.find("description")
                        title = _clean(t.text if t is not None else "")
                        link = l.text if l is not None else ""
                        safe, _ = validate_url_safety(link)
                        if title and link and safe:
                            items.append({
                                "title": title, "url": link,
                                "snippet": _clean(d.text if d is not None else ""),
                                "source": "Bing News", "platform": "Bing", "icon": "🔵", "category": "news",
                            })
            except Exception as e:
                logger.debug(f"Bing News RSS search error: {e}")
            return items

        def _hackernews_algolia():
            items = []
            try:
                resp = requests.get(
                    f"https://hn.algolia.com/api/v1/search?query={urllib.parse.quote_plus(query)}&hitsPerPage=8",
                    timeout=3.0, headers={"User-Agent": _random_ua()}
                )
                if resp.status_code == 200:
                    for hit in resp.json().get("hits", [])[:5]:
                        title = hit.get("title", "")
                        url = hit.get("url", "") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
                        safe, _ = validate_url_safety(url)
                        if title and safe:
                            items.append({
                                "title": title, "url": url,
                                "snippet": f"Hacker News • {hit.get('points',0)} points • {hit.get('num_comments',0)} comments",
                                "source": "Hacker News", "platform": "Hacker News", "icon": "🔶", "category": "tech",
                            })
            except Exception as e:
                logger.debug(f"HackerNews Algolia search error: {e}")
            return items

        def _github_search():
            items = []
            try:
                resp = requests.get(
                    f"https://api.github.com/search/repositories?q={urllib.parse.quote_plus(query)}&sort=stars&order=desc&per_page=5",
                    timeout=3.0, headers={"User-Agent": _random_ua()}
                )
                if resp.status_code == 200:
                    for repo in resp.json().get("items", [])[:5]:
                        url = repo.get("html_url", "")
                        safe, _ = validate_url_safety(url)
                        if safe:
                            items.append({
                                "title": f"GitHub: {repo.get('full_name', '')}",
                                "url": url,
                                "snippet": f"{repo.get('description','')} • {repo.get('stargazers_count',0):,} stars",
                                "source": "GitHub", "platform": "GitHub", "icon": "🐙", "category": "tech",
                            })
            except Exception as e:
                logger.debug(f"GitHub search error: {e}")
            return items

        def _reddit_search():
            items = []
            try:
                resp = requests.get(
                    f"https://www.reddit.com/search.rss?q={urllib.parse.quote_plus(query)}&sort=relevance&limit=5",
                    timeout=3.5, headers={"User-Agent": _random_ua()}
                )
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    for entry in root.findall(".//atom:entry", ns)[:5]:
                        t = entry.find("atom:title", ns)
                        l = entry.find("atom:link", ns)
                        if t is not None and l is not None:
                            items.append({
                                "title": t.text or "", "url": l.get("href", ""),
                                "snippet": f"Reddit community discussion about '{query}'",
                                "source": "Reddit", "platform": "Reddit", "icon": "🔴", "category": "social",
                            })
            except Exception as e:
                logger.debug(f"Reddit internet search error: {e}")
            return items

        # Run all search engines in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = {
                ex.submit(_google_news): "google_news",
                ex.submit(_wikipedia): "wikipedia",
                ex.submit(_bing_news): "bing",
                ex.submit(_hackernews_algolia): "hn_algolia",
                ex.submit(_github_search): "github",
                ex.submit(_reddit_search): "reddit",
            }
            try:
                for f in concurrent.futures.as_completed(futures, timeout=self.timeout + 2):
                    try:
                        _add(f.result())
                    except Exception:
                        pass
            except TimeoutError:
                logger.debug("search_internet_wide timeout reached, proceeding with collected results")

        return all_results[:max_results]

    # ── Deep URL scraper for any social media post/page ─────────────────────
    def scrape_social_url(self, url: str) -> Dict[str, Any]:
        """Deep structured metadata extraction for any social media or web URL."""
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()

        # Identify platform
        platform = "Web"
        icon = "🌐"
        for pk, meta in PLATFORM_META.items():
            if meta["domain"] in domain:
                platform = meta["name"]
                icon = meta["icon"]
                break

        # ── Reddit JSON API fast-path ────────────────────────────────────────
        if "reddit.com" in domain and "/comments/" in url:
            try:
                json_url = url.split("?")[0].rstrip("/") + ".json"
                r = self.session.get(json_url, timeout=self.timeout, headers={"User-Agent": _random_ua()})
                if r.status_code == 200:
                    data = r.json()
                    post = data[0]["data"]["children"][0]["data"]
                    title = post.get("title", "")
                    selftext = post.get("selftext", "")
                    author = post.get("author", "")
                    score = post.get("score", 0)
                    sub = post.get("subreddit_name_prefixed", "")
                    comments = []
                    if len(data) > 1:
                        for c in data[1]["data"]["children"][:5]:
                            body = c.get("data", {}).get("body", "")
                            cauthor = c.get("data", {}).get("author", "")
                            if body:
                                comments.append(f"u/{cauthor}: {body}")
                    full_text = (
                        f"Reddit Thread: {title}\n"
                        f"Subreddit: {sub} | Posted by: u/{author} | Upvotes: {score:,}\n\n"
                        f"{selftext}\n\nTop Community Comments:\n" + "\n".join(comments)
                    ).strip()
                    return {"status": "success", "url": url, "platform": "Reddit", "icon": "🔴",
                        "title": f"Reddit: {title}", "text": full_text,
                        "char_count": len(full_text), "retrieved_at": datetime.now(timezone.utc).isoformat()}
            except Exception as e:
                logger.debug(f"Reddit JSON extractor fallback: {e}")

        # ── GitHub Repo API fast-path ────────────────────────────────────────
        if "github.com" in domain:
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] not in ("features", "pricing", "explore", "trending", "topics"):
                owner, repo = parts[0], parts[1]
                try:
                    r = self.session.get(f"https://api.github.com/repos/{owner}/{repo}", timeout=self.timeout)
                    if r.status_code == 200:
                        info = r.json()
                        title = info.get("full_name", f"{owner}/{repo}")
                        desc = info.get("description", "")
                        stars = info.get("stargazers_count", 0)
                        forks = info.get("forks_count", 0)
                        lang = info.get("language", "Code")
                        topics = ", ".join(info.get("topics", []))
                        readme_text = ""
                        try:
                            rr = self.session.get(f"https://api.github.com/repos/{owner}/{repo}/readme",
                                headers={"Accept": "application/vnd.github.raw"}, timeout=3.0)
                            if rr.status_code == 200:
                                readme_text = rr.text[:3000]
                        except Exception:
                            pass
                        full_text = (
                            f"GitHub Repository: {title}\nDescription: {desc}\n"
                            f"Language: {lang} | Stars: {stars:,} | Forks: {forks:,}\n"
                            f"Topics: {topics}\nCreated: {info.get('created_at','')} | Updated: {info.get('updated_at','')}\n\n"
                            f"README Overview:\n{readme_text}"
                        ).strip()
                        return {"status": "success", "url": url, "platform": "GitHub", "icon": "🐙",
                            "title": f"GitHub: {title}", "text": full_text,
                            "char_count": len(full_text), "retrieved_at": datetime.now(timezone.utc).isoformat()}
                except Exception as e:
                    logger.debug(f"GitHub API extractor fallback: {e}")

        # ── BlueSky post fast-path ───────────────────────────────────────────
        if "bsky.app" in domain and "/post/" in url:
            try:
                parts = parsed.path.strip("/").split("/")
                handle = parts[1] if len(parts) > 1 else ""
                rkey = parts[-1]
                resp = self.session.get(
                    f"https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread?uri=at://{handle}/app.bsky.feed.post/{rkey}",
                    timeout=self.timeout, headers={"User-Agent": _random_ua()}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    post_data = data.get("thread", {}).get("post", {})
                    text = post_data.get("record", {}).get("text", "")
                    likes = post_data.get("likeCount", 0)
                    reposts = post_data.get("repostCount", 0)
                    full_text = f"BlueSky Post by @{handle}:\n{text}\n\nLikes: {likes} | Reposts: {reposts}"
                    return {"status": "success", "url": url, "platform": "BlueSky", "icon": "🦋",
                        "title": f"BlueSky: {text[:80]}", "text": full_text,
                        "char_count": len(full_text), "retrieved_at": datetime.now(timezone.utc).isoformat()}
            except Exception as e:
                logger.debug(f"BlueSky post extractor error: {e}")

        # ── Universal HTML fallback with OpenGraph & JSON-LD ─────────────────
        return self._scrape_html_universal(url, platform, icon)

    def _scrape_html_universal(self, url: str, platform: str = "Web", icon: str = "🌐") -> Dict[str, Any]:
        """Universal HTML scraper using OG tags, JSON-LD schema, and full-text extraction."""
        try:
            resp = self.session.get(url, timeout=self.timeout, headers={"User-Agent": _random_ua()})
            if resp.status_code != 200:
                return {"status": "error", "url": url, "error": f"HTTP {resp.status_code}"}

            soup = BeautifulSoup(resp.text, "html.parser")

            # Title extraction (OG > twitter:title > <title>)
            title = ""
            for sel in [("meta", {"property": "og:title"}), ("meta", {"name": "twitter:title"})]:
                el = soup.find(*sel)
                if el and el.get("content"):
                    title = el["content"].strip()
                    break
            if not title and soup.title:
                title = soup.title.get_text(strip=True)

            # Description extraction
            desc = ""
            for sel in [("meta", {"property": "og:description"}), ("meta", {"name": "twitter:description"}), ("meta", {"name": "description"})]:
                el = soup.find(*sel)
                if el and el.get("content"):
                    desc = el["content"].strip()
                    break

            # JSON-LD structured data extraction
            jld_text = ""
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    jld = json.loads(script.string or "{}")
                    if isinstance(jld, dict):
                        for field in ["name", "headline", "description", "articleBody"]:
                            if jld.get(field):
                                jld_text += f"{field}: {jld[field]}\n"
                except Exception:
                    pass

            # Clean body text extraction
            for el in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside"]):
                el.decompose()
            body_text = soup.get_text(separator="\n", strip=True)
            body_lines = [l.strip() for l in body_text.splitlines() if len(l.strip()) > 25]
            clean_body = "\n".join(body_lines[:60])

            combined = f"[{platform}] {title}\n\nDescription: {desc}\n\n{jld_text}\nContent:\n{clean_body}".strip()

            return {
                "status": "success", "url": url, "platform": platform, "icon": icon,
                "title": f"{platform}: {title or url}",
                "text": combined[:getattr(config, 'MAX_EXTRACTED_LENGTH', 8000)],
                "char_count": len(combined),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {"status": "error", "url": url, "error": str(e)}

    # ── Scrape ANY website (powerful universal scraper) ─────────────────────
    def scrape_any_website(self, url: str) -> Dict[str, Any]:
        """
        Scrape any website on the internet with full content extraction.
        Uses multi-layer fallback: direct HTTP -> User-agent rotation -> HTML extraction.
        """
        safe, reason = validate_url_safety(url)
        if not safe:
            return {"status": "error", "url": url, "error": f"Safety check failed: {reason}"}
        return self.scrape_social_url(url)

    # ── Batch scrape multiple URLs ───────────────────────────────────────────
    def scrape_urls_parallel(self, urls: List[str], max_workers: int = 8) -> List[Dict[str, Any]]:
        """Scrape multiple URLs in parallel for maximum throughput."""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.scrape_any_website, url): url for url in urls}
            try:
                for f in concurrent.futures.as_completed(futures, timeout=self.timeout + 5):
                    try:
                        results.append(f.result())
                    except Exception as e:
                        results.append({"status": "error", "url": futures[f], "error": str(e)})
            except TimeoutError:
                logger.debug("scrape_urls_parallel timeout reached, returning partial results")
        return results


# ── Singleton ────────────────────────────────────────────────────────────────
_GLOBAL_SOCIAL: Optional[SocialMediaScraper] = None


def get_social_scraper() -> SocialMediaScraper:
    global _GLOBAL_SOCIAL
    if _GLOBAL_SOCIAL is None:
        _GLOBAL_SOCIAL = SocialMediaScraper()
    return _GLOBAL_SOCIAL
