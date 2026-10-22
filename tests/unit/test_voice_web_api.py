"""Unit tests for Voice Web API endpoints (/api/voice and /api/voice/control)."""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app


class TestVoiceWebAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_voice_empty_transcript_400(self):
        res = self.client.post("/api/voice", json={"transcript": "   "})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_voice_fast_path_stop(self):
        res = self.client.post("/api/voice", json={"transcript": "stop"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("type"), "control_fast_path")
        self.assertEqual(data.get("action"), "EMERGENCY_STOP")

    def test_voice_control_mute(self):
        res = self.client.post("/api/voice/control", json={"action": "mute", "muted": True})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("is_muted"))

        # Unmute
        res_unmute = self.client.post("/api/voice/control", json={"action": "mute", "muted": False})
        self.assertEqual(res_unmute.status_code, 200)
        self.assertFalse(res_unmute.get_json().get("is_muted"))

    def test_voice_control_barge_in(self):
        res = self.client.post("/api/voice/control", json={"action": "barge_in"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("action"), "barge_in")
        self.assertIn("latency_ms", data)

    def test_voice_control_status(self):
        res = self.client.post("/api/voice/control", json={"action": "status"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("state", data)
        self.assertIn("is_muted", data)

    def test_voice_conversational_utterance(self):
        res = self.client.post("/api/voice", json={"transcript": "Hello Qwen, how are you today?"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("type"), "conversational_turn")
        self.assertIn("spoken_text", data)
        self.assertIn("display_text", data)


if __name__ == "__main__":
    unittest.main()
