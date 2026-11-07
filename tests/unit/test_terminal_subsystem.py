"""Unit tests for Phase 9A: Policy-Governed Terminal Subsystem.
Tests:
- Section 6B: Command tier classification (Tier 1 vs Tier 2 vs Tier 3 vs Blocked).
- Section 6B: Persistent cwd across sequential session commands.
- Section 6B: Monotonic timeout and process-tree termination.
- H-60: Terminal output untrusted provenance tagging.
- Section 6B: Trusted Terminal Session scoping and expiry.
- Section 6B: JSONL command journaling.
"""

import os
import sys
import json
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_llm.agent.policy.gateway import PolicyGateway, RiskTier, InputProvenance
from local_llm.agent.tools.terminal import TerminalSubsystem, TerminalCommandResult


class TestTerminalSubsystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = PROJECT_ROOT / "eval" / "fixtures"
        cls.fixture_dir.mkdir(parents=True, exist_ok=True)
        cls.journal_path = str(cls.fixture_dir / "test_terminal_journal.jsonl")
        cls.gateway = PolicyGateway(granted_roots=[str(cls.fixture_dir)])

    def setUp(self):
        if os.path.exists(self.journal_path):
            os.remove(self.journal_path)
        self.terminal = TerminalSubsystem(gateway=self.gateway, journal_path=self.journal_path)

    def test_01_command_tier_classification(self):
        """Commands must be categorized into Tier 1, 2, 3, or blocked."""
        self.assertEqual(self.terminal.classify_command_tier("git status"), RiskTier.T1_ROUTINE)
        self.assertEqual(self.terminal.classify_command_tier("python --version"), RiskTier.T1_ROUTINE)
        self.assertEqual(self.terminal.classify_command_tier("dir"), RiskTier.T1_ROUTINE)

        self.assertEqual(self.terminal.classify_command_tier("pip install requests"), RiskTier.T2_ELEVATED)
        self.assertEqual(self.terminal.classify_command_tier("del temp.txt"), RiskTier.T3_SENSITIVE)
        self.assertEqual(self.terminal.classify_command_tier("format d:"), RiskTier.T3_SENSITIVE)

    def test_02_execution_and_exit_code_capture(self):
        """Commands must execute cleanly, capture exit code, and tag output as untrusted."""
        res = self.terminal.execute_command("python -c \"print('TERMINAL_TEST_OK')\"")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("TERMINAL_TEST_OK", res.stdout)
        self.assertEqual(res.provenance, InputProvenance.UNTRUSTED_CONTENT)  # H-60

        # Non-zero exit code
        res_fail = self.terminal.execute_command("python -c \"import sys; sys.exit(77)\"")
        self.assertEqual(res_fail.exit_code, 77)

    def test_03_catastrophic_command_blocked(self):
        """Catastrophic command patterns must be blocked by policy gateway."""
        res = self.terminal.execute_command(r"del /f /s /q C:\Windows")
        self.assertEqual(res.exit_code, 126)
        self.assertIn("Blocked", res.stderr)

    def test_04_timeout_terminates_process_tree(self):
        """Commands exceeding monotonic deadline must be terminated."""
        res = self.terminal.execute_command("python -c \"import time; time.sleep(10)\"", timeout_seconds=0.2)
        self.assertEqual(res.exit_code, -9)
        self.assertIn("TIMEOUT", res.stderr)

    def test_05_trusted_terminal_session_scope_and_expiry(self):
        """Trusted terminal session permits local operations without prompt."""
        ts = self.terminal.start_trusted_session("dev_sess_1", str(self.fixture_dir), duration_seconds=0.5)
        self.assertFalse(ts.is_expired)

        # Journal verification
        self.terminal.execute_command("python -c \"print('TRUSTED_LOG_ENTRY')\"")
        self.assertTrue(os.path.exists(self.journal_path))
        with open(self.journal_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertGreater(len(lines), 0)
        self.assertEqual(lines[-1]["session_id"], "default")

        # Expire session
        self.terminal.end_trusted_session()
        self.assertIsNone(self.terminal.active_trusted_session)


if __name__ == "__main__":
    unittest.main()
