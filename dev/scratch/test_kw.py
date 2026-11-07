import urllib.request
import json
import re

def is_keyword_research_query(query: str) -> bool:
    q = query.lower()
    kw_indicators = [
        'search volume', 'keyword difficulty', 'keyword tool', 'keyword search tool',
        'cpc of', 'monthly searches', 'seo volume', 'ranking difficulty'
    ]
    if any(k in q for k in kw_indicators):
        return True
    if re.search(r'\bkd\b', q) and any(w in q for w in ['keyword', 'search', 'find', 'volume', 'google', 'tool', 'seo']):
        return True
    return False

def extract_target_keyword(query: str) -> str:
    quoted = re.findall(r'["\']([^"\']+)["\']', query)
    if quoted:
        return quoted[0].strip()
    pat = re.compile(r'(?:search\s+volume|kd|keyword\s+difficulty).*?(?:of|for)\s+(?:the\s+)?([^\?\.!\n]+)', re.I)
    m = pat.search(query)
    if m:
        kw = m.group(1).strip()
        kw = re.sub(r'^(?:the|a|an)\s+', '', kw, flags=re.I)
        return kw
    return query.strip()

test_q = 'find from the google or any keyword search tool the search volume and KD of the produces and reviews'
print('Is KW query:', is_keyword_research_query(test_q))
print('Extracted KW:', extract_target_keyword(test_q))
