"""Viral Script Generation Engine.
Fetches real-time trending data then generates fully-structured viral scripts
for TikTok, YouTube Shorts, Instagram Reels, LinkedIn posts, X threads,
podcasts, and long-form YouTube videos.

Workflow:
  1. Fetch trending topics relevant to the requested topic/query
  2. Gather live social media and news context from the internet
  3. Build a rich script context dossier
  4. Return structured script data ready for LLM prompting
"""

import concurrent.futures
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger("script_engine")

# Script format templates per platform
SCRIPT_FORMAT_TEMPLATES = {
    "tiktok": {
        "name": "TikTok / Instagram Reels",
        "icon": "🎵",
        "duration": "15-60 seconds",
        "structure": [
            "🎬 HOOK (0-3 sec): Stop-the-scroll opening line + visual action",
            "😱 SHOCK/INTRIGUE (3-8 sec): Unexpected fact or bold claim",
            "📊 THE FACTS (8-35 sec): Core verified information, fast-paced",
            "💥 THE TWIST (35-45 sec): Surprising reveal or climax",
            "📢 CTA (45-60 sec): Comment-bait question or challenge",
            "🏷️ HASHTAGS: 6-10 trending hashtags + viral caption",
        ],
        "tone": "fast-paced, punchy, emotional, viral-optimized",
    },
    "youtube_shorts": {
        "name": "YouTube Shorts",
        "icon": "▶️",
        "duration": "30-60 seconds",
        "structure": [
            "🎬 HOOK (0-3 sec): Verbal + visual hook that demands attention",
            "⚡ VALUE PROMISE (3-8 sec): Tell them exactly what they'll learn",
            "📖 STORY/FACTS (8-45 sec): Crisp storytelling with B-roll cues",
            "🎯 PUNCHLINE (45-55 sec): The satisfying conclusion or reveal",
            "🔔 CTA (55-60 sec): Subscribe + comment prompt",
        ],
        "tone": "educational-entertaining (edutainment), engaging, clear",
    },
    "linkedin": {
        "name": "LinkedIn Post / Article",
        "icon": "💼",
        "duration": "Post: 300-800 words",
        "structure": [
            "🔑 HOOK LINE: Bold contrarian statement or striking statistic",
            "📌 CONTEXT: Why this matters to professionals right now",
            "💡 3-5 KEY INSIGHTS: Bullet points with data and examples",
            "🔮 IMPLICATIONS: What this means for your career/industry",
            "💬 ENGAGEMENT QUESTION: Invite discussion and debate",
            "🏷️ HASHTAGS: 3-5 professional hashtags",
        ],
        "tone": "professional, authoritative, thought-leadership",
    },
    "x_thread": {
        "name": "X (Twitter) Thread",
        "icon": "𝕏",
        "duration": "7-15 tweets",
        "structure": [
            "Tweet 1 - HOOK: Bold claim or mind-blowing stat (280 chars)",
            "Tweets 2-3 - CONTEXT: Background and why it matters",
            "Tweets 4-8 - THE BREAKDOWN: Numbered facts with evidence",
            "Tweet 9-10 - THE TWIST: Unexpected angle or counterpoint",
            "Tweet 11 - CONCLUSION: Key takeaway summarized",
            "Tweet 12 - CTA: Retweet/follow prompt + question",
        ],
        "tone": "direct, sharp, opinion-driven, shareable",
    },
    "youtube_long": {
        "name": "YouTube Long-Form Video",
        "icon": "🎬",
        "duration": "8-15 minutes",
        "structure": [
            "🎬 COLD OPEN (0-30s): Most dramatic moment or bold question",
            "📣 INTRO (30s-2min): Topic intro + subscribe prompt",
            "📚 SECTION 1 (2-5min): Background and context with B-roll",
            "💥 SECTION 2 (5-9min): The main revelation or analysis",
            "🔮 SECTION 3 (9-12min): Implications and expert opinions",
            "📢 OUTRO (12-15min): Summary + CTA + next video tease",
        ],
        "tone": "documentary-style, engaging, in-depth, journalistic",
    },
    "podcast": {
        "name": "Podcast Script",
        "icon": "🎙️",
        "duration": "10-30 minutes",
        "structure": [
            "🎙️ COLD OPEN: Teaser clip of the most interesting moment",
            "👋 INTRO SEGMENT: Host intro + topic overview (2 min)",
            "📖 ACT 1: Setting the scene and background (5 min)",
            "🔍 ACT 2: Deep dive with evidence and expert quotes (10 min)",
            "💬 ACT 3: Analysis, debate, different perspectives (8 min)",
            "🎯 OUTRO: Key takeaways + listener CTA + credits",
        ],
        "tone": "conversational, informative, engaging, narrative-driven",
    },
}


class ScriptEngine:
    """Fetches trending data and builds viral script generation contexts."""

    def __init__(self):
        self._trending_engine = None
        self._social_scraper = None

    @property
    def trending_engine(self):
        if self._trending_engine is None:
            from trending_engine import get_trending_engine
            self._trending_engine = get_trending_engine()
        return self._trending_engine

    @property
    def social_scraper(self):
        if self._social_scraper is None:
            from social_scraper import get_social_scraper
            self._social_scraper = get_social_scraper()
        return self._social_scraper

    def detect_script_format(self, prompt: str) -> str:
        """Auto-detect the target script format from the user prompt."""
        p_lower = prompt.lower()
        if any(k in p_lower for k in ["youtube short", "youtube shorts", "yt short", "yt shorts"]):
            return "youtube_shorts"
        if any(k in p_lower for k in ["tiktok", "reels", "reel", "instagram"]):
            return "tiktok"
        if any(k in p_lower for k in ["linkedin", "professional post"]):
            return "linkedin"
        if any(k in p_lower for k in ["twitter thread", "x thread", "tweet thread", "x post", "tweet"]):
            return "x_thread"
        if any(k in p_lower for k in ["youtube video", "long form", "long-form", "youtube long", "full video"]):
            return "youtube_long"
        if any(k in p_lower for k in ["podcast", "audio script", "episode"]):
            return "podcast"
        # Default to TikTok (most viral format)
        return "tiktok"

    def extract_topic(self, prompt: str) -> str:
        """Extract the core script topic from the prompt."""
        topic = prompt
        clean_patterns = [
            r"^(?:please\s+)?(?:can\s+you\s+)?(?:write|create|make|generate|produce)\s+(?:me\s+)?(?:a\s+)?(?:viral\s+)?(?:high-retention\s+)?(?:video\s+|tiktok\s+|youtube\s+shorts\s+|youtube\s+|reels\s+|podcast\s+|linkedin\s+|x\s+|twitter\s+)?script\s+(?:on|about|for|regarding|on\s+this|on\s+trending)?\s*",
            r"\b(?:write\s+(?:a\s+)?(?:viral\s+)?script|create\s+(?:a\s+)?(?:viral\s+)?script|make\s+(?:a\s+)?(?:viral\s+)?script)\b",
            r"\b(?:tiktok\s+script|youtube\s+shorts\s+script|youtube\s+script|reels\s+script|podcast\s+script|video\s+script)\b",
        ]
        for pat in clean_patterns:
            topic = re.sub(pat, "", topic, flags=re.IGNORECASE)
        topic = re.sub(r"^(?:on|about|for|regarding|this)\s+", "", topic.strip(), flags=re.IGNORECASE)
        return topic.strip(" .,;:-?!\"'") or prompt

    def gather_script_context(self, topic: str, script_format: str) -> Dict[str, Any]:
        """
        Gather real-time trending context and social media data for the script topic.
        Returns a structured context dict with trending topics, news, and social data.
        """
        context = {
            "topic": topic,
            "script_format": script_format,
            "format_info": SCRIPT_FORMAT_TEMPLATES.get(script_format, SCRIPT_FORMAT_TEMPLATES["tiktok"]),
            "trending_hooks": [],
            "news_context": [],
            "social_reactions": [],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        def _get_trends():
            try:
                return self.trending_engine.fetch_trending_for_script(topic_hint=topic, limit=5)
            except Exception as e:
                logger.debug(f"Script trending fetch error: {e}")
                return []

        def _get_social():
            try:
                result = self.social_scraper.search_all_social_media(
                    topic,
                    platforms=["tiktok", "x", "youtube", "reddit", "instagram"],
                    limit_per_platform=2,
                )
                return result.get("results", [])
            except Exception as e:
                logger.debug(f"Script social fetch error: {e}")
                return []

        def _get_news():
            try:
                return self.social_scraper.search_internet_wide(topic, max_results=6)
            except Exception as e:
                logger.debug(f"Script news fetch error: {e}")
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            fut_trends = ex.submit(_get_trends)
            fut_social = ex.submit(_get_social)
            fut_news = ex.submit(_get_news)
            try:
                context["trending_hooks"] = fut_trends.result()
            except Exception:
                pass
            try:
                context["social_reactions"] = fut_social.result()
            except Exception:
                pass
            try:
                context["news_context"] = fut_news.result()
            except Exception:
                pass

        return context

    def build_script_prompt_context(self, prompt: str) -> Dict[str, Any]:
        """
        Main entry: detect format, gather live data, return full script context
        for LLM injection.
        """
        script_format = self.detect_script_format(prompt)
        topic = self.extract_topic(prompt)
        context = self.gather_script_context(topic, script_format)
        fmt_info = context["format_info"]

        # Build the text dossier for LLM injection
        lines = [
            f"=== REAL-TIME SCRIPT RESEARCH DOSSIER ===",
            f"Topic: {topic}",
            f"Target Platform: {fmt_info['name']} ({fmt_info['duration']})",
            f"Tone: {fmt_info['tone']}",
            f"Fetched At: {context['fetched_at']}",
            "",
            "=== VIRAL TRENDING HOOKS FROM INTERNET ===",
        ]

        if context["trending_hooks"]:
            for i, t in enumerate(context["trending_hooks"], 1):
                lines.append(f"{i}. [{t['platform']}] {t['title']} ({t['metric']})")
                lines.append(f"   Context: {t['snippet']}")
                lines.append(f"   URL: {t['url']}")
        else:
            lines.append("(Using topic-specific research below)")

        lines.append("\n=== LIVE NEWS & INTERNET RESEARCH ===")
        if context["news_context"]:
            for item in context["news_context"][:5]:
                lines.append(f"- [{item.get('source','Web')}] {item.get('title','')} — {item.get('snippet','')[:150]}")
                lines.append(f"  URL: {item.get('url','')}")
        else:
            lines.append("(No live news context available for this topic)")

        lines.append("\n=== SOCIAL MEDIA SENTIMENT & VIRAL REACTIONS ===")
        if context["social_reactions"]:
            for item in context["social_reactions"][:6]:
                lines.append(f"- [{item.get('platform','Social')}] {item.get('title','')} — {item.get('snippet','')[:120]}")
        else:
            lines.append("(No social media reactions found for this topic)")

        lines.append("\n=== SCRIPT STRUCTURE TO FOLLOW ===")
        for sec in fmt_info["structure"]:
            lines.append(f"  {sec}")

        lines.append("=== END OF SCRIPT RESEARCH DOSSIER ===")

        dossier_text = "\n".join(lines)

        return {
            "topic": topic,
            "script_format": script_format,
            "format_info": fmt_info,
            "dossier_text": dossier_text,
            "context": context,
        }


# ── Singleton ────────────────────────────────────────────────────────────────
_GLOBAL_SCRIPT_ENGINE: Optional[ScriptEngine] = None


def get_script_engine() -> ScriptEngine:
    global _GLOBAL_SCRIPT_ENGINE
    if _GLOBAL_SCRIPT_ENGINE is None:
        _GLOBAL_SCRIPT_ENGINE = ScriptEngine()
    return _GLOBAL_SCRIPT_ENGINE
