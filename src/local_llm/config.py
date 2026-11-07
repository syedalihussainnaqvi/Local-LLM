"""Centralized configuration for the local AI chatbot with web-scraping & RAG capabilities."""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = str(PROJECT_ROOT / "models" / "adapter")
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", PROJECT_ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(os.environ.get("RAG_DB_PATH", DATA_DIR / "rag_storage.db"))

# Model Configuration
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

# Web Access & Scraping Settings
WEB_ACCESS_ENABLED = os.environ.get("WEB_ACCESS_ENABLED", "true").lower() in ("true", "1", "yes")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "5.0"))  # fast 5s per request
MAX_RESPONSE_SIZE = int(os.environ.get("MAX_RESPONSE_SIZE", str(5 * 1024 * 1024)))  # 5 MB max
MAX_REDIRECTS = int(os.environ.get("MAX_REDIRECTS", "5"))
USER_AGENT = os.environ.get(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (ResearchBot/1.0; LocalAI)"
)
REQUEST_DELAY = float(os.environ.get("REQUEST_DELAY", "0.2"))  # polite delay between requests

# Security: Allowed schemes and forbidden IP subnets
ALLOWED_SCHEMES = ("http", "https")
BLOCKED_PORTS = {22, 23, 25, 445, 3389, 6379, 27017, 9200}

# Text Processing & Chunking
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "600"))  # characters per chunk
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "120"))  # characters overlap
MAX_EXTRACTED_LENGTH = int(os.environ.get("MAX_EXTRACTED_LENGTH", "150000"))  # max chars per doc

# Retrieval Settings
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.25"))
MAX_CONTEXT_LENGTH = int(os.environ.get("MAX_CONTEXT_LENGTH", "3500"))  # chars in prompt context

# Web Discovery Settings: 'authentic_composite', 'google_news', 'wikipedia', 'duckduckgo', 'brave', 'mock'
SEARCH_PROVIDER = os.environ.get("SEARCH_PROVIDER", "authentic_composite")
MAX_SEARCH_RESULTS = int(os.environ.get("MAX_SEARCH_RESULTS", "5"))
MAX_URLS_PER_QUERY = int(os.environ.get("MAX_URLS_PER_QUERY", "5"))
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "")

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
