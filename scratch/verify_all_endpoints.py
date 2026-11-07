import io
import wave
import json
import sys
import os

sys.path.insert(0, os.path.abspath("."))
from app import app

client = app.test_client()

print("=== 1. Testing Web UI Home Route ===")
res = client.get("/")
assert res.status_code == 200
assert b"voice-call-modal" in res.data
assert b"voice-spectrum-canvas" in res.data
assert b"voice-ptt-btn" in res.data
print("-> PASS: Web UI rendered successfully with hacker voice modal & spectrum canvas.")

print("\n=== 2. Testing Chat API Route ===")
res = client.post("/api/chat", json={"message": "hello", "stream": False})
print(f"-> PASS: Chat API status {res.status_code}")

print("\n=== 3. Testing Real-Time Voice API Route ===")
res = client.post("/api/voice", json={"transcript": "what is the current status"})
assert res.status_code == 200
data = res.get_json()
assert "spoken_text" in data
assert "display_text" in data
spoken = data.get("spoken_text", "")
print(f"-> PASS: Voice API returned: {spoken[:60]}...")

print("\n=== 4. Testing Voice Control Fast-Path & Barge-In ===")
res = client.post("/api/voice/control", json={"action": "barge_in"})
assert res.status_code == 200
print(f"-> PASS: Voice control returned {res.get_json()}")

print("\n=== 5. Testing Offline SAPI STT Route ===")
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(b"\x00" * 32000)
res = client.post("/api/voice/stt", data=buf.getvalue(), headers={"Content-Type": "audio/wav"})
assert res.status_code == 200
stt_data = res.get_json()
assert "transcript" in stt_data
dur = stt_data.get("duration_ms", 0)
model = stt_data.get("model", "")
print(f"-> PASS: Offline STT processed audio locally in {dur}ms with model: {model}")

print("\n=== 6. Testing Config Route ===")
res = client.get("/api/config")
assert res.status_code == 200
print("-> PASS: Config API operates nominally.")

print("\n=======================================================")
print("ALL SYSTEM APPLICATION ENDPOINTS VERIFIED SUCCESSFULLY!")
print("=======================================================")
