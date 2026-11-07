import sys
import os
import json
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath("."))

from web_scraper import SafeWebScraper, validate_url_safety, SecurityException
from social_scraper import get_social_scraper, PLATFORM_META
import trending_engine
from app import app

def run_tests():
    print("=" * 70)
    print("LIVE REAL-TIME SCRAPING & WEB RETRIEVAL COMPREHENSIVE TEST")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. SECURITY & SSRF BOUNDARY CHECK
    # -------------------------------------------------------------
    print("\n[TEST 1] SSRF & Private IP Protection Guard")
    blocked_targets = [
        "http://127.0.0.1:5000/api/config",
        "http://localhost:8080/secret",
        "http://169.254.169.254/latest/meta-data/",
        "http://192.168.1.1/admin",
    ]
    for url in blocked_targets:
        safe, reason = validate_url_safety(url)
        print(f"  Target: {url}")
        print(f"  Result: {'BLOCKED (PASS)' if not safe else 'ALLOWED (FAIL)'} | Reason: {reason}")
        assert not safe, f"Security failure: {url} should have been blocked"

    # -------------------------------------------------------------
    # 2. DIRECT UNIVERSAL WEB SCRAPING (REAL LIVE WEB PAGE)
    # -------------------------------------------------------------
    print("\n[TEST 2] Universal Live Web Scraping")
    scraper = SafeWebScraper()
    test_urls = [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://docs.python.org/3/",
    ]

    for url in test_urls:
        print(f"\n  Fetching real page: {url}")
        t0 = time.time()
        res = scraper.scrape(url)
        dur = round((time.time() - t0) * 1000, 2)
        status = res.get("status")
        title = res.get("title", "")
        char_count = res.get("char_count", 0)
        snippet = res.get("text", "")[:250].replace("\n", " ")

        print(f"  -> Status: {status} (HTTP {res.get('http_status')}) in {dur}ms")
        print(f"  -> Title: {title}")
        print(f"  -> Extracted Characters: {char_count:,} chars")
        print(f"  -> Actual Content Snippet:\n     \"{snippet}...\"")
        assert status == "success", f"Scraping failed for {url}: {res.get('error')}"
        assert char_count > 100, "Insufficient extracted text"

    # -------------------------------------------------------------
    # 3. SOCIAL MEDIA SCRAPING - GITHUB API
    # -------------------------------------------------------------
    print("\n[TEST 3] Platform Scraper: GitHub Search")
    soc = get_social_scraper()
    gh_query = "autonomous coding agent python"
    print(f"  Query: '{gh_query}'")
    t0 = time.time()
    gh_results = soc.search_single_platform("github", gh_query, limit=3)
    dur = round((time.time() - t0) * 1000, 2)
    print(f"  -> Returned {len(gh_results)} live repositories in {dur}ms:")
    for idx, r in enumerate(gh_results, 1):
        print(f"     [{idx}] {r.get('title')} | URL: {r.get('url')}")
        print(f"         Snippet: {r.get('snippet')}")
    assert len(gh_results) > 0, "GitHub search returned 0 results"

    # -------------------------------------------------------------
    # 4. SOCIAL MEDIA SCRAPING - REDDIT & GOOGLE NEWS RSS
    # -------------------------------------------------------------
    print("\n[TEST 4] Platform Scraper: Reddit & News Feeds")
    rd_query = "artificial intelligence breakthroughs"
    print(f"  Query: '{rd_query}' on Reddit")
    rd_results = soc.search_single_platform("reddit", rd_query, limit=3)
    print(f"  -> Returned {len(rd_results)} Reddit posts:")
    for idx, r in enumerate(rd_results, 1):
        print(f"     [{idx}] {r.get('title')} | Source: {r.get('source')}")
        print(f"         URL: {r.get('url')}")

    # -------------------------------------------------------------
    # 5. MULTI-PLATFORM PARALLEL SOCIAL MEDIA SEARCH
    # -------------------------------------------------------------
    print("\n[TEST 5] Multi-Platform Simultaneous Social Search")
    target_platforms = ["github", "reddit", "youtube", "hackernews"]
    multi_query = "deepseek llm"
    print(f"  Query: '{multi_query}' across platforms: {target_platforms}")
    t0 = time.time()
    multi_res = soc.search_all_social_media(multi_query, platforms=target_platforms, limit_per_platform=2)
    dur = round((time.time() - t0) * 1000, 2)
    flat_results = multi_res.get("results", [])
    print(f"  -> Total Aggregated Results: {len(flat_results)} across {len(multi_res.get('by_platform', {}))} platforms in {dur}ms")
    for r in flat_results[:6]:
        print(f"     [{r.get('icon')} {r.get('platform')}] {r.get('title')[:60]}... -> {r.get('url')}")

    # -------------------------------------------------------------
    # 6. INTERNET-WIDE SEARCH (Google News RSS / Multi-source)
    # -------------------------------------------------------------
    print("\n[TEST 6] Internet-Wide Search Engine")
    net_query = "quantum computing milestone 2026"
    print(f"  Query: '{net_query}'")
    t0 = time.time()
    net_results = soc.search_internet_wide(net_query, max_results=4)
    dur = round((time.time() - t0) * 1000, 2)
    print(f"  -> Returned {len(net_results)} internet-wide search hits in {dur}ms:")
    for idx, item in enumerate(net_results, 1):
        print(f"     [{idx}] {item.get('title')} ({item.get('source', 'Web')})")
        print(f"         URL: {item.get('url')}")

    # -------------------------------------------------------------
    # 7. REAL-TIME TRENDING TOPICS SCRAPER
    # -------------------------------------------------------------
    print("\n[TEST 7] Live Trending Topics Engine")
    from trending_engine import get_trending_engine
    t0 = time.time()
    engine = get_trending_engine()
    trends = engine.get_all_trending_topics(category="all", limit=6)
    dur = round((time.time() - t0) * 1000, 2)
    print(f"  -> Fetched {len(trends)} trending topics in {dur}ms:")
    for idx, t in enumerate(trends[:6], 1):
        print(f"     [{idx}] {t.get('icon', '🔥')} {t.get('title', t.get('topic'))} ({t.get('platform', t.get('source'))}) - {t.get('metric', '')}")

    # -------------------------------------------------------------
    # 8. FLASK API SCRAPING ENDPOINTS END-TO-END
    # -------------------------------------------------------------
    print("\n[TEST 8] Flask API Scraping Endpoints Integration")
    client = app.test_client()

    # /api/social-search
    api_res = client.post("/api/social-search", json={"query": "python fastapi", "platforms": ["github", "reddit"], "limit": 2})
    assert api_res.status_code == 200
    api_data = api_res.get_json()
    print(f"  -> /api/social-search status 200 | count: {api_data.get('count', 0)} items")

    # /api/trending
    trend_res = client.get("/api/trending?limit=5")
    assert trend_res.status_code == 200
    trend_data = trend_res.get_json()
    print(f"  -> /api/trending status 200 | count: {trend_data.get('count', 0)} trends")

    print("\n" + "=" * 70)
    print("ALL SCRAPING MODULES TESTED WITH REAL LIVE WEB RESPONSES: SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
