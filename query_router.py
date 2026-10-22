"""Intelligent query routing module for local LLM, local RAG, direct URL, and automatic live web retrieval."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import config
from retrieval import HybridRetriever, get_retriever
from storage import StorageManager, get_storage

logger = logging.getLogger("query_router")

# Regex to detect raw URLs directly in user prompt
URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

# Trivial greetings/farewells that don't need web access
GREETING_REGEX = re.compile(
    r"^(hi|hello|hey|greetings|howdy|good\s*(morning|afternoon|evening|day)|thanks|thank you|bye|goodbye)[!.? ]*$",
    re.IGNORECASE,
)


def is_trending_query(prompt: str) -> bool:
    """Detect if query asks for trending topics, viral stories, or hot internet trends."""
    p_lower = prompt.lower()
    trending_keywords = [
        "trending topic", "trending topics", "what is trending", "what's trending",
        "fetch trending", "trending on", "viral today", "viral topics",
        "hot topics", "trending now", "trending stories", "top trends",
        "trending on tiktok", "trending on twitter", "trending on youtube",
        "trending on x", "trending on reddit", "trending repositories",
        "trending news", "viral trends"
    ]
    return any(k in p_lower for k in trending_keywords)


def is_scriptwriting_query(prompt: str) -> bool:
    """Detect if query asks for a video/social/viral script on a topic or trend."""
    p_lower = prompt.lower()
    script_keywords = [
        "write script", "write a script", "create a script", "video script",
        "tiktok script", "shorts script", "youtube script", "reels script",
        "viral script", "podcast script", "content script", "script on this",
        "script on trending", "script about"
    ]
    return any(k in p_lower for k in script_keywords)


def is_social_search_query(prompt: str) -> bool:
    """Detect if query asks to search social media platforms (TikTok, X, Instagram, YouTube, LinkedIn, GitHub, Reddit, Facebook)."""
    p_lower = prompt.lower()
    social_platforms = [
        "tiktok", "instagram", "twitter", "x twitter", "youtube",
        "linkedin", "github", "facebook", "reddit", "social media",
        "pinterest", "twitch", "medium", "bluesky", "mastodon",
        "snapchat", "quora", "tumblr", "telegram", "steam",
        "product hunt", "producthunt", "hacker news", "hackernews"
    ]
    has_platform = any(sp in p_lower for sp in social_platforms)
    has_search_action = any(act in p_lower for act in [
        "search", "scrape", "find", "look up", "discussions", "posts",
        "what people say", "crawl", "data on", "search on"
    ])
    return has_platform and has_search_action


def is_verification_query(prompt: str) -> bool:
    """Detect if query asks to cross-verify news, claims, or rumors across platforms."""
    p_lower = prompt.lower()
    verify_keywords = [
        "verify", "fact check", "fact-check", "is it true", "is that true",
        "verify each news", "from different platforms", "cross verify",
        "cross-check", "debunk", "rumor", "fake news", "confirm if",
        "check if this is true", "verify this claim", "verify news"
    ]
    return any(vk in p_lower for vk in verify_keywords)


# Coding / Generative task detection (fast-path for code, math, and writing tasks without temporal need)
CODING_OR_GENERATIVE_KEYWORDS = [
    "write code", "python", "javascript", "typescript", "html", "css", "c++", "c#",
    "java", "rust", "golang", "sql", "react", "vue", "angular", "flask", "django",
    "fastapi", "dockerfile", "bash script", "powershell", "regex", "algorithm",
    "function", "class", "debug", "refactor", "unit test",
    "create an app", "create a website", "build a", "implement", "binary search",
    "linked list", "fibonacci", "rest api", "crud", "frontend", "backend",
    "write an essay", "write a story", "write a poem", "solve", "calculate",
    "how to code", "programming", "code for"
]


def is_math_logic_or_reasoning(prompt: str) -> bool:
    """
    Detect if query is a pure math, number comparison, logic puzzle, or reasoning question
    that should be solved by the LLM directly without web scraping or knowledge base retrieval.
    Examples:
      - 'which is greater 9.9 or 9.11'
      - 'compare 9.9 and 9.11'
      - '9.9 vs 9.11'
      - 'is 15 greater than 20'
      - 'what is 25 * 4'
      - 'solve 2x + 5 = 15'
      - 'how many r in strawberry'
    """
    p = prompt.lower().strip()

    # 1. Number comparison patterns (e.g. which is greater 9.9 or 9.11, 9.9 vs 9.11)
    comparison_patterns = [
        r"\bwhich\s+is\s+(?:greater|bigger|smaller|larger|higher|lower|more|less)\b",
        r"\b(?:is|are)\s+[-+]?\d*\.?\d+\s+(?:greater|bigger|smaller|larger|higher|lower|more|less)\b",
        r"\bcompare\s+[-+]?\d*\.?\d+\s+(?:and|with|to|or)\s+[-+]?\d*\.?\d+\b",
        r"[-+]?\d*\.?\d+\s*(?:>|<|>=|<=|==)\s*[-+]?\d*\.?\d+",
        r"\b(?:greater|bigger|smaller|larger)\s+(?:than|or)\b",
        r"\b[-+]?\d*\.?\d+\s+(?:or|vs|versus)\s+[-+]?\d*\.?\d+\b",
        r"\b(?:max|min|maximum|minimum)\s*\(\s*[-+]?\d*\.?\d+",
    ]
    for pat in comparison_patterns:
        if re.search(pat, p):
            return True

    # 2. Arithmetic, algebra, and math calculations
    math_patterns = [
        r"^\s*[-+]?\d*\.?\d+\s*[\+\-\*\/\^\%]\s*[-+]?\d*\.?\d+",
        r"\b(?:what\s+is|calculate|evaluate|solve)\s+[-+]?\d*\.?\d+\s*[\+\-\*\/\^\%]",
        r"\b(?:square\s+root|sqrt|factorial|prime\s+number|fibonacci)\b",
        r"\b(?:solve|equation)\s+[0-9a-z\s\+\-\*\/\=\^\(\)]+\s*=",
    ]
    for pat in math_patterns:
        if re.search(pat, p):
            return True

    # 3. Logic, riddles, letter counting
    logic_phrases = [
        "how many r", "how many letters", "how many words", "riddle", "logic puzzle",
        "strawberry", "if a plane crashes", "which weighs more"
    ]
    if any(phrase in p for phrase in logic_phrases):
        return True

    return False


def is_desktop_action_query(prompt: str) -> bool:
    """
    Detect if query requests autonomous desktop, mouse, keyboard, app, file, or terminal control.
    """
    p = prompt.lower().strip()
    action_keywords = [
        "open app", "close app", "open apps", "close apps", "launch app",
        "open whatsapp", "close whatsapp", "whatsapp and send message", "send message", "delete message",
        "open chrome", "open google", "scroll page", "open chatgpt", "give task",
        "run command", "on terminal", "in terminal", "powershell", "cmd command", "terminal command",
        "click mouse", "move mouse", "press key", "type text",
        "create file", "delete file", "edit file", "modify file",
        "full charge of my laptop", "control mouse", "control keyboard",
        "desktop agent", "automate desktop", "take control of laptop",
        "take charge of my laptop", "full control of my laptop"
    ]
    return any(kw in p for kw in action_keywords)


def is_temporal_or_web_needed(prompt: str) -> bool:
    """Check if query requires live web access (trending, news, sports, live rates, explicit web search)."""
    p_lower = prompt.lower()
    
    # 0. Trending topics, Social Media search, Video Scriptwriting, and Verification
    if (
        is_trending_query(prompt)
        or is_social_search_query(prompt)
        or is_verification_query(prompt)
        or (is_scriptwriting_query(prompt) and any(w in p_lower for w in ["trend", "news", "viral", "this", "today", "topic", "ai", "tech", "event"]))
    ):
        return True

    # 1. Keyword research & SEO metrics
    if any(k in p_lower for k in [
        "search volume", "kd of", "keyword difficulty", "semrush", "ahrefs",
        "google keyword", "keyword planner", "keywords for", "volume and kd"
    ]):
        return True
        
    # 2. Currency & live rates
    if any(c in p_lower for c in [
        "dollar price", "exchange rate", "usd to", "rate today", "current price",
        "price of dollar", "price of usd", "gold price", "currency rate"
    ]):
        return True
        
    # 3. News & Sports
    if any(w in p_lower for w in [
        "news", "headline", "breaking news", "cricket score", "football score",
        "match score", "tournament update", "standings", "who won the match"
    ]):
        return True
        
    # 4. Explicit web search phrases (require explicit user request to browse or search online)
    web_search_phrases = [
        "search google for", "search the web for", "search online for", "search internet for",
        "google search for", "find online", "find on google", "lookup online",
        "search on google", "browse web", "browse online",
        "search data on every single website"
    ]
    if any(phrase in p_lower for phrase in web_search_phrases):
        return True
        
    # 5. Temporal triggers for current state of the world
    words = set(re.findall(r"\b[a-z0-9_]+\b", p_lower))
    temporal_triggers = {
        "today", "latest", "current", "currently", "weather", "forecast", "trending"
    }
    if words & temporal_triggers:
        # Ignore coding/algorithmic expressions like "current node", "current directory", "current index"
        coding_phrases = ["current node", "current pointer", "current index", "current directory", "current element", "current value"]
        if any(cp in p_lower for cp in coding_phrases):
            return False
        return True
        
    return False

def is_coding_or_generative_task(prompt: str) -> bool:
    """Detect if query is primarily a programming, coding, debugging, or creative writing task."""
    p_lower = prompt.lower()
    # If the user is asking for trending, social media, verification, or script on a trend, do NOT treat as offline coding!
    if is_trending_query(prompt) or is_social_search_query(prompt) or is_verification_query(prompt):
        return False
    if is_scriptwriting_query(prompt):
        return False
    return any(trigger in p_lower for trigger in CODING_OR_GENERATIVE_KEYWORDS)


class QueryRouter:
    """Routes user queries to LLM-only, Local Knowledge, Live Web, or Direct URL handling."""

    def __init__(self, storage: Optional[StorageManager] = None, retriever: Optional[HybridRetriever] = None):
        self.storage = storage or get_storage()
        self.retriever = retriever or get_retriever()

    def extract_urls(self, text: str) -> List[str]:
        """Extract valid http/https URLs from prompt text."""
        return URL_PATTERN.findall(text)

    def route_query(
        self,
        prompt: str,
        user_override_mode: str = "auto",
        web_access_enabled: bool = config.WEB_ACCESS_ENABLED,
    ) -> Dict[str, Any]:
        """
        Evaluate query routing.
        Automatically routes questions to live web when web access is enabled,
        with seamless offline fallback.
        """
        cleaned_prompt = prompt.strip()
        extracted_urls = self.extract_urls(cleaned_prompt)

        # 1. Direct URL in prompt (e.g. "Summarize https://example.com" or just "https://...")
        if extracted_urls:
            return {
                "route": "direct_url",
                "extracted_urls": extracted_urls,
                "reason": f"Detected direct URL(s): {', '.join(extracted_urls)}",
                "search_query": cleaned_prompt,
            }

        mode = (user_override_mode or "auto").lower()

        # 2. Explicit User Overrides
        if mode == "llm_only":
            return {
                "route": "llm_only",
                "extracted_urls": [],
                "reason": "User explicitly selected LLM Only mode",
                "search_query": cleaned_prompt,
            }

        if mode == "local_rag":
            return {
                "route": "local_rag",
                "extracted_urls": [],
                "reason": "User explicitly selected Local Knowledge mode",
                "search_query": cleaned_prompt,
            }

        if mode == "live_web":
            if not web_access_enabled:
                return {
                    "route": "local_rag",
                    "extracted_urls": [],
                    "reason": "Live Web requested but Web Access is currently disabled; falling back to Local Knowledge",
                    "search_query": cleaned_prompt,
                }
            return {
                "route": "live_web",
                "extracted_urls": [],
                "reason": "User explicitly selected Live Web mode",
                "search_query": cleaned_prompt,
            }

        # 3. Automatic Mode ("auto")
        # Check if there is an active pending desktop action plan awaiting user authorization
        from desktop_agent import get_desktop_agent
        desktop_agent = get_desktop_agent()
        pending_plan = desktop_agent.get_pending_plan()
        if pending_plan:
            if desktop_agent.is_approval_intent(cleaned_prompt):
                return {
                    "route": "desktop_action_execute",
                    "plan_id": pending_plan.plan_id,
                    "extracted_urls": [],
                    "reason": f"User explicitly approved execution of Action Plan {pending_plan.plan_id}",
                    "search_query": cleaned_prompt,
                }
            elif desktop_agent.is_rejection_intent(cleaned_prompt):
                return {
                    "route": "desktop_action_reject",
                    "plan_id": pending_plan.plan_id,
                    "extracted_urls": [],
                    "reason": f"User explicitly rejected Action Plan {pending_plan.plan_id}",
                    "search_query": cleaned_prompt,
                }

        # Check if it's purely a greeting / conversational pleasantry
        if GREETING_REGEX.match(cleaned_prompt):
            return {
                "route": "llm_only",
                "extracted_urls": [],
                "reason": "Conversational greeting; answering directly with local model",
                "search_query": cleaned_prompt,
            }

        # 0. Autonomous Desktop & Computer Use Actions
        if is_desktop_action_query(cleaned_prompt):
            return {
                "route": "desktop_action",
                "extracted_urls": [],
                "reason": "Autonomous desktop/OS action requested (apps, mouse/keyboard, terminal, files); requesting user permission before execution",
                "search_query": cleaned_prompt,
            }

        # 1. Pure Math, Number Comparisons & Logic Reasoning Fast-Path
        # Never scrape or search local knowledge for basic math, number comparisons ("which is greater 9.9 or 9.11"), or logic puzzles.
        if is_math_logic_or_reasoning(cleaned_prompt):
            return {
                "route": "llm_only",
                "extracted_urls": [],
                "reason": "Pure math, number comparison, or logic reasoning; answering directly with local LLM without unnecessary web scraping",
                "search_query": cleaned_prompt,
            }

        # 2. FAST-PATH FOR CODING, ALGORITHMS & GENERATIVE TASKS:
        # If the user asks for code, programming, debugging, math, or writing without needing temporal/web data,
        # route directly to llm_only for INSTANT response generation with zero scraping latency.
        if is_coding_or_generative_task(cleaned_prompt) and not is_temporal_or_web_needed(cleaned_prompt):
            return {
                "route": "llm_only",
                "extracted_urls": [],
                "reason": "Programming or generative request; answering with full Claude-grade code intelligence (instant streaming)",
                "search_query": cleaned_prompt,
            }

        # 3. Live Web retrieval when current events, temporal data, news, sports, keyword metrics, or live search are needed
        if web_access_enabled and is_temporal_or_web_needed(cleaned_prompt):
            return {
                "route": "live_web",
                "extracted_urls": [],
                "reason": "Automatically retrieving current authentic information from web sources",
                "search_query": cleaned_prompt,
            }

        # 4. Check if local knowledge base has highly relevant documents (threshold >= 0.65)
        stats = self.storage.get_stats()
        if stats["active_sources"] > 0:
            relevant_chunks = self.retriever.search(
                cleaned_prompt,
                top_k=1,
                threshold=0.65,
            )
            if relevant_chunks:
                top_score = relevant_chunks[0].get("score", 0.0)
                return {
                    "route": "local_rag",
                    "extracted_urls": [],
                    "reason": f"Found relevant local knowledge passages (top score: {top_score:.2f})",
                    "search_query": cleaned_prompt,
                }

        # Default fallback: answer directly with existing local LLM
        return {
            "route": "llm_only",
            "extracted_urls": [],
            "reason": "General question; answering directly with fine-tuned local model",
            "search_query": cleaned_prompt,
        }


# Singleton accessor
_GLOBAL_ROUTER: Optional[QueryRouter] = None

def get_router() -> QueryRouter:
    global _GLOBAL_ROUTER
    if _GLOBAL_ROUTER is None:
        _GLOBAL_ROUTER = QueryRouter()
    return _GLOBAL_ROUTER
