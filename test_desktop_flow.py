import urllib.request, json

req = urllib.request.Request(
    'http://127.0.0.1:5000/api/chat',
    data=json.dumps({'prompt': 'open some apps that are present in my laptop like google chrome whatspp microsoft store file exprorer search something on google the live mouse moving typing is shown on the app'}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
with urllib.request.urlopen(req) as resp:
    for line in resp:
        line_str = line.decode('utf-8').strip()
        if line_str.startswith('data: '):
            ev = json.loads(line_str[6:])
            if 'desktop_plan' in ev:
                p = ev['desktop_plan']
                print(f"SUCCESS! Plan generated: {p['plan_id']} with {len(p['steps'])} steps across {len(p['phases'])} phases.")
                for idx, s in enumerate(p['steps'], 1):
                    print(f"  {idx}. [{s['phase']}] [{s['action_type']}] {s['description']}")
            elif 'token' in ev:
                print("Notice Token:", ev['token'])
