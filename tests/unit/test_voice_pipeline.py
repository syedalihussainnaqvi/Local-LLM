"""Unit tests for Phase 9B: Voice Call Mode Pipeline.
Tests:
- H-56: Spoken channel normalization (code, URLs, markdown).
- H-58: Deterministic control fast-path (stop, pause, resume, mute).
- C-50 / C-51: Voice safety check and mandatory non-voice confirmation for Tier 2+/Tier 3.
- H-57: Mute toggle provably stops incoming audio frame processing.
- H-54: Barge-in interruption latency (target <= 250ms).
- H-53: Sentence-chunked streaming.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_llm.agent.policy.gateway import RiskTier
from local_llm.agent.voice.pipeline import (
    SpeechNormalizer,
    VoiceControlFastPath,
    VoiceSafetyValidator,
    VoicePipeline,
    VoiceTurnState,
)


class TestVoicePipeline(unittest.TestCase):

    def test_01_speech_normalizer_spoken_channel(self):
        """H-56: Normalizer separates display and spoken channels, converting code and markdown."""
        raw_markdown = (
            "# Task Complete\n\n"
            "Here is the script you requested:\n"
            "```python\n"
            "def calculate_total(a, b):\n"
            "    return a + b\n"
            "```\n"
            "You can view more details at https://example.com/docs.\n"
            "**Note**: Ensure parameters are positive."
        )
        channels = SpeechNormalizer.normalize(raw_markdown)

        # Spoken text must not read code or raw URLs
        self.assertNotIn("def calculate_total", channels.spoken_text)
        self.assertNotIn("https://example.com", channels.spoken_text)
        self.assertIn("displayed the code", channels.spoken_text)
        self.assertNotIn("#", channels.spoken_text)
        self.assertNotIn("**", channels.spoken_text)

        # Display text retains full markdown
        self.assertEqual(channels.display_text, raw_markdown.strip())

    def test_02_control_phrases_fast_path(self):
        """H-58: Control phrases must resolve immediately via fast path."""
        self.assertEqual(VoiceControlFastPath.evaluate("stop"), "EMERGENCY_STOP")
        self.assertEqual(VoiceControlFastPath.evaluate("Stop!"), "EMERGENCY_STOP")
        self.assertEqual(VoiceControlFastPath.evaluate("freeze"), "EMERGENCY_STOP")
        self.assertEqual(VoiceControlFastPath.evaluate("pause"), "PAUSE")
        self.assertEqual(VoiceControlFastPath.evaluate("wait"), "PAUSE")
        self.assertEqual(VoiceControlFastPath.evaluate("resume"), "RESUME")
        self.assertEqual(VoiceControlFastPath.evaluate("mute"), "MUTE")

        # Non-control query returns None
        self.assertIsNone(VoiceControlFastPath.evaluate("What is the weather today?"))

    def test_03_voice_safety_non_voice_confirmation(self):
        """C-50, C-51: Sensitive commands require non-voice confirmation and readback."""
        # Tier 1: Routine app launch
        allowed_t1, prompt_t1 = VoiceSafetyValidator.inspect_voice_intent(RiskTier.T1_ROUTINE, "open notepad")
        self.assertTrue(allowed_t1)
        self.assertIsNone(prompt_t1)

        # Tier 3: Deletion / system mutate
        allowed_t3, prompt_t3 = VoiceSafetyValidator.inspect_voice_intent(RiskTier.T3_SENSITIVE, "delete important_file.txt")
        self.assertFalse(allowed_t3)
        self.assertIsNotNone(prompt_t3)
        self.assertIn("confirm button", prompt_t3.lower())

    def test_04_mute_provably_drops_frames(self):
        """H-57: Mute toggle must reject incoming audio frame processing."""
        pipeline = VoicePipeline()
        dummy_frame = b"\x00" * 320  # 10ms of 16kHz mono audio

        # Unmuted
        self.assertTrue(pipeline.process_incoming_audio_frame(dummy_frame))

        # Muted
        pipeline.set_mute(True)
        self.assertFalse(pipeline.process_incoming_audio_frame(dummy_frame))

        # Unmuted again
        pipeline.set_mute(False)
        self.assertTrue(pipeline.process_incoming_audio_frame(dummy_frame))

    def test_05_barge_in_latency(self):
        """H-54: Barge-in stops speaking state within <=250ms budget."""
        pipeline = VoicePipeline()
        pipeline.state = VoiceTurnState.SPEAKING

        stop_latency = pipeline.handle_barge_in()
        self.assertEqual(pipeline.state, VoiceTurnState.INTERRUPTED)
        self.assertLessEqual(stop_latency, 250.0)

    def test_06_sentence_chunking_stream(self):
        """H-53: Token stream chunks at sentence punctuation."""
        def token_generator():
            tokens = ["The ", "research ", "report ", "is ", "ready. ", "I ", "saved ", "it ", "to ", "Documents."]
            for t in tokens:
                yield t

        chunks = list(VoicePipeline.chunk_sentences(token_generator()))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "The research report is ready.")
        self.assertEqual(chunks[1], "I saved it to Documents.")


if __name__ == "__main__":
    unittest.main()
