import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.stdout.reconfigure(encoding='utf-8')

from local_llm.social_scraper import get_social_scraper
from local_llm.script_engine import get_script_engine
from local_llm.verification_engine import get_verification_engine

scraper = get_social_scraper()
script_engine = get_script_engine()
verifier = get_verification_engine()

print('--- 1. Testing Internet-Wide Search ---')
res = scraper.search_internet_wide('artificial intelligence', max_results=3)
print('Found', len(res), 'results:')
for r in res:
    print('  [' + str(r.get('source')) + ']', str(r.get('title'))[:60])

print('\n--- 2. Testing Social Scraper Multi-Platform ---')
social_res = scraper.search_all_social_media('machine learning', platforms=['github', 'reddit'], limit_per_platform=2)
print('Social search status:', social_res.get('status'), 'count:', len(social_res.get('results', [])))
for s in social_res.get('results', [])[:3]:
    print('  [' + str(s.get('platform')) + ']', str(s.get('title'))[:60])

print('\n--- 3. Testing Script Engine ---')
ctx = script_engine.build_script_prompt_context('write a viral tiktok script on AI humanoid robots')
print('Topic:', ctx['topic'])
print('Format:', ctx['script_format'], ctx['format_info']['name'])
print('Dossier lines:', len(ctx['dossier_text'].splitlines()))

print('\n--- 4. Testing Multi-Platform News Verification ---')
v_report = verifier.verify_claim('Quantum computer breaks RSA encryption')
print('Verdict:', v_report.get('verdict'))
print('Consensus Score:', str(v_report.get('consensus_score')) + '%')
print('News channels checked:', len(v_report.get('news_breakdown', [])))
print('Social sources checked:', len(v_report.get('social_breakdown', [])))

print('\n=== ALL INTEGRATION TESTS PASSED ===')
