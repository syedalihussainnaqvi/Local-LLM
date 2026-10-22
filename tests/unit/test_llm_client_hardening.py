"""Unit tests for Phase 2: LLM Client Hardening.
Tests:
- Context injection (H-24)
- Monotonic deadlines and cancellation tokens (H-25)
- Structured output with reasoning-first field order (H-31)
- Unified tool-call format and round-trip parsing (H-32)
- Structured flight logging to JSONL (M-38)
"""

import os
import sys
import time
import json
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.llm_client import (
    ContextInjector,
    ToolCallParser,
    ToolCall,
    MonotonicDeadline,
    CancellationToken,
    HardenedLLMClient,
    StructuredOutput,
)


class TestLLMClientHardening(unittest.TestCase):

    def test_01_context_injector(self):
        """H-24: System prompt must contain current date, time, OS, and cutoff."""
        block = ContextInjector.get_context_block()
        self.assertIn("[SYSTEM CONTEXT - AUTHORITATIVE]", block)
        self.assertIn("Host OS:", block)
        self.assertIn("Date:", block)
        self.assertIn("Model Knowledge Cutoff: September 2024", block)
        self.assertIn("[END SYSTEM CONTEXT]", block)

    def test_02_tool_call_formatting_and_parsing_roundtrip(self):
        """H-32: Single unified tool-call format with round-trip fidelity."""
        name = "launch_browser"
        args = {"query": "deep learning 2026", "headless": False}
        reasoning = "User asked for recent deep learning news."

        formatted = ToolCallParser.format_tool_call(name=name, arguments=args, reasoning=reasoning)
        self.assertIn("<tool_call>", formatted)
        self.assertIn("</tool_call>", formatted)

        parsed = ToolCallParser.parse_tool_call(formatted)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.name, name)
        self.assertEqual(parsed.arguments, args)
        self.assertEqual(parsed.reasoning, reasoning)

    def test_03_markdown_and_embedded_json_parsing(self):
        """Parser must handle markdown fences and raw embedded JSON gracefully."""
        md_text = (
            "I will open notepad for you:\n"
            "```json\n"
            "{\n"
            "  \"reasoning\": \"Opening notepad as requested\",\n"
            "  \"name\": \"open_app\",\n"
            "  \"arguments\": {\"app\": \"notepad\"}\n"
            "}\n"
            "```"
        )
        parsed = ToolCallParser.parse_tool_call(md_text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.name, "open_app")
        self.assertEqual(parsed.arguments["app"], "notepad")
        self.assertEqual(parsed.reasoning, "Opening notepad as requested")

    def test_04_monotonic_deadline(self):
        """H-25: Monotonic deadlines must expire accurately without wall-clock drift."""
        deadline = MonotonicDeadline(timeout_seconds=0.1)
        self.assertFalse(deadline.is_expired)
        self.assertGreater(deadline.remaining_seconds, 0.0)

        time.sleep(0.12)
        self.assertTrue(deadline.is_expired)
        self.assertEqual(deadline.remaining_seconds, 0.0)

    def test_05_cancellation_token(self):
        """Tokens must allow immediate cancellation across threads."""
        token = CancellationToken()
        self.assertFalse(token.is_cancelled)
        token.cancel()
        self.assertTrue(token.is_cancelled)

    def test_06_hardened_client_generation_and_flight_logging(self):
        """M-38: Client must produce structured output and append JSONL flight log."""
        test_log = PROJECT_ROOT / "eval" / "fixtures" / "test_flight_log.jsonl"
        if test_log.exists():
            test_log.unlink()

        client = HardenedLLMClient(engine=None, log_path=str(test_log))
        out = client.generate(
            prompt="Open Google Chrome",
            task_id="task_test_h2",
            step_id="step_test_h2",
            timeout_seconds=5.0,
        )

        self.assertIsInstance(out, StructuredOutput)
        self.assertEqual(out.decision, "open_app")
        self.assertIsNotNone(out.tool_call)
        self.assertEqual(out.tool_call.arguments["app"], "chrome")

        # Verify flight log
        self.assertTrue(test_log.exists())
        with open(test_log, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(lines), 1)
        record = lines[0]
        self.assertEqual(record["task_id"], "task_test_h2")
        self.assertEqual(record["decision"], "open_app")
        self.assertTrue(record["tool_call_detected"])


if __name__ == "__main__":
    unittest.main()
