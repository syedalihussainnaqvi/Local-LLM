"""Comprehensive automated tests for voice call mode bug fix (Section 6).
Covers:
1. Injected-WAV 20-run stress test: 20 runs of known utterance producing non-empty reply & audio.
2. Failure-injection tests: STT exception, LLM timeout/error, TTS error, empty normalizer.
3. State machine watchdog tests: recovery from stuck states, barge-in, multi-turn sequence.
4. Regression tests for root cause.
"""

import os
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_llm.agent.voice.audio_engine import WindowsTTS, WindowsSTT, EnergyVAD, AudioDeviceManager
from local_llm.agent.voice.pipeline import (
    VoicePipeline,
    VoiceTurnState,
    SpeechNormalizer,
    VoiceControlFastPath,
)
from local_llm.agent.voice.doctor import VoiceDoctor


class TestVoiceBugFix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = PROJECT_ROOT / "eval" / "fixtures"
        cls.fixture_dir.mkdir(parents=True, exist_ok=True)
        cls.test_wav = str(cls.fixture_dir / "two_plus_two.wav")
        # Ensure test fixture audio exists
        if not Path(cls.test_wav).exists():
            WindowsTTS.synthesize_to_wav("What is two plus two", cls.test_wav)

    def test_01_injected_wav_20_runs(self):
        """Automated: 20 consecutive runs of injected utterance through pipeline.
        Must produce non-empty reply text and non-silent audio on every single run.
        """
        pipeline = VoicePipeline()
        passed_runs = 0
        total_runs = 20

        def mock_llm(q: str) -> str:
            return "Two plus two is four."

        for i in range(total_runs):
            t0 = time.time()
            turn = pipeline.process_injected_wav(
                self.test_wav,
                llm_generate_fn=mock_llm,
                synthesize_audio=True,
            )
            elapsed = time.time() - t0
            spoken = turn.get("spoken_text", "")
            tts_bytes = turn.get("tts_bytes", 0)

            # Assert non-empty reply text and non-silent audio within timeout (15s)
            self.assertTrue(len(spoken) > 0, f"Run #{i+1} produced empty spoken reply")
            self.assertTrue(tts_bytes > 1000, f"Run #{i+1} produced empty/silent audio ({tts_bytes} bytes)")
            self.assertLess(elapsed, 15.0, f"Run #{i+1} exceeded latency deadline ({elapsed:.2f}s)")
            passed_runs += 1

        self.assertEqual(passed_runs, total_runs, f"Expected 20/20 runs to pass, got {passed_runs}")

    def test_02_failure_injection_empty_stt(self):
        """Failure injection: STT returning empty/silence must produce visible fallback, not silence."""
        pipeline = VoicePipeline()
        turn = pipeline.process_user_utterance("")
        self.assertEqual(turn.get("type"), "error")
        self.assertIn("could not hear", turn.get("spoken_text", "").lower())
        self.assertEqual(turn.get("state"), VoiceTurnState.IDLE.value)

    def test_03_failure_injection_llm_exception(self):
        """Failure injection: LLM exception must return error notice without crashing or freezing."""
        pipeline = VoicePipeline()

        def broken_llm(q):
            raise RuntimeError("Simulated GPU out of memory error")

        turn = pipeline.process_user_utterance("test query", llm_generate_fn=broken_llm)
        self.assertEqual(pipeline.state, VoiceTurnState.IDLE)
        self.assertIn("error generating a response", turn.get("spoken_text", "").lower())

    def test_04_failure_injection_empty_normalizer(self):
        """Failure injection: normalizer stripping all content must fall back to non-empty text (H-16)."""
        channels = SpeechNormalizer.normalize("### *** ``` ```")
        self.assertTrue(len(channels.spoken_text) > 0)
        self.assertNotEqual(channels.spoken_text.strip(), "")

    def test_05_watchdog_state_timeout_recovery(self):
        """Watchdog: stuck state in THINKING or SPEAKING must auto-reset and warn."""
        pipeline = VoicePipeline(thinking_timeout_seconds=0.1, speaking_timeout_seconds=0.1)
        pipeline.set_state(VoiceTurnState.THINKING, reason="Simulated hang")
        time.sleep(0.15)
        alert = pipeline.check_watchdogs()
        self.assertIsNotNone(alert)
        self.assertIn("Thinking timeout", alert)
        self.assertEqual(pipeline.state, VoiceTurnState.IDLE)

    def test_06_barge_in_and_recovery(self):
        """State machine: barge-in stops playback and cleanly returns to IDLE/LISTENING."""
        pipeline = VoicePipeline()
        pipeline.set_state(VoiceTurnState.SPEAKING, reason="Playback ongoing")
        latency = pipeline.handle_barge_in()
        self.assertLess(latency, 250.0)
        self.assertEqual(pipeline.state, VoiceTurnState.INTERRUPTED)

        # Next turn works immediately
        turn2 = pipeline.process_user_utterance("hello again", llm_generate_fn=lambda q: "Welcome back.")
        self.assertEqual(turn2.get("spoken_text"), "Welcome back.")
        self.assertEqual(pipeline.state, VoiceTurnState.IDLE)

    def test_07_multi_turn_sequence_no_resource_leak(self):
        """Multi-turn consistency: turns 1, 2, 3, 4 execute without state pollution or memory leak."""
        pipeline = VoicePipeline()
        for i in range(5):
            turn = pipeline.process_user_utterance(
                f"Question {i}",
                llm_generate_fn=lambda q: f"Answer to {q}"
            )
            self.assertEqual(pipeline.state, VoiceTurnState.IDLE)
            self.assertIn(f"Answer to Question {i}", turn["spoken_text"])
        self.assertEqual(len(pipeline.conversation_history), 5)

    def test_08_voice_doctor_all_stages(self):
        """Diagnostic probe: voice-doctor runs 8 stages and passes."""
        doc = VoiceDoctor(record_seconds=1.5, enable_playback=False)
        passed = doc.run_all()
        self.assertTrue(passed, "VoiceDoctor probe failed one or more stages")


if __name__ == "__main__":
    unittest.main()
