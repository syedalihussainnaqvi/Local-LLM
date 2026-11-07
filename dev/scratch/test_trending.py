import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.stdout.reconfigure(encoding='utf-8')
from local_llm.trending_engine import get_trending_engine

engine = get_trending_engine()
trends = engine.get_all_trending_topics(limit=8)
print(f"Successfully fetched {len(trends)} trending topics across the web:")
for idx, t in enumerate(trends, 1):
    print(f"{idx}. {t['icon']} [{t['platform']}] {t['title']} ({t['metric']})")
