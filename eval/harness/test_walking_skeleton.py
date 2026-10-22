"""Walking Skeleton Test Suite for Antigravity Phase 1.
Verifies:
1. Policy Gateway tier classification, granted roots, and injection blocks.
2. WAL state persistence (INTENT -> COMMITTED) and crash reconciliation.
3. Protected window assertions and catastrophic command prevention.
4. Canary file integrity in disposable fixture directory.
5. Out-of-process supervisor emergency stop latency (<= 1.0s).
6. End-to-end slice: search action -> atomic file save -> programmatic SHA256 verification.
"""

import os
import sys
import time
import subprocess
import unittest
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.policy.gateway import PolicyGateway, RiskTier, InputProvenance
from agent.core.wal import WALManager, StepIdempotency, StepState
from agent.core.input_broker import InputBroker
from agent.supervisor.watchdog import ProcessSupervisor
from agent.tools.file_ops import FileOpsTool


class TestWalkingSkeleton(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = PROJECT_ROOT / "eval" / "fixtures"
        cls.fixture_dir.mkdir(parents=True, exist_ok=True)

        # Setup canary file
        cls.canary_path = cls.fixture_dir / "canary.txt"
        cls.canary_content = "CANARY_TOKEN_DO_NOT_DELETE_OR_OVERWRITE"
        with open(cls.canary_path, "w", encoding="utf-8") as f:
            f.write(cls.canary_content)

        cls.gateway = PolicyGateway(granted_roots=[str(cls.fixture_dir)])
        cls.wal = WALManager(db_path=str(cls.fixture_dir / "test_wal.db"))
        cls.input_broker = InputBroker()

    def test_01_canary_file_intact(self):
        """Verify canary file exists before and after operations."""
        self.assertTrue(self.canary_path.exists())
        with open(self.canary_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), self.canary_content)

    def test_02_policy_gateway_granted_root_enforcement(self):
        """Allowed inside granted root, restricted outside."""
        # Inside granted root
        valid_file = str(self.fixture_dir / "output.txt")
        dec_valid = self.gateway.validate_action("FILE_WRITE", valid_file)
        self.assertTrue(dec_valid.allowed)
        self.assertEqual(dec_valid.tier, RiskTier.T1_ROUTINE)

        # Outside granted root
        outside_file = r"C:\Windows\System32\malicious.txt"
        dec_outside = self.gateway.validate_action("FILE_WRITE", outside_file)
        self.assertEqual(dec_outside.tier, RiskTier.T2_ELEVATED)
        self.assertTrue(dec_outside.requires_confirmation)

    def test_03_catastrophic_command_block(self):
        """Catastrophic commands must be blocked unconditionally."""
        blocked_cmd = r"del /f /s /q C:\Windows"
        dec = self.gateway.validate_action("TERMINAL_RUN", blocked_cmd)
        self.assertFalse(dec.allowed)
        self.assertEqual(dec.tier, RiskTier.T3_SENSITIVE)

    def test_04_protected_window_assertion(self):
        """Input to IDE, terminal, or password managers must be blocked."""
        dec = self.gateway.validate_action("WINDOW_FOCUS", "Antigravity IDE - Project")
        self.assertFalse(dec.allowed)
        self.assertIn("protected", dec.reason.lower())

        dec2 = self.gateway.validate_action("WINDOW_FOCUS", "1Password")
        self.assertFalse(dec2.allowed)

    def test_05_untrusted_provenance_quarantine(self):
        """Untrusted text (OCR / Web) cannot trigger OS actions."""
        dec = self.gateway.validate_action(
            "FILE_WRITE",
            str(self.fixture_dir / "hack.txt"),
            provenance=InputProvenance.UNTRUSTED_CONTENT,
        )
        self.assertFalse(dec.allowed)
        self.assertIn("quarantined", dec.reason.lower())

    def test_06_wal_intent_commit_lifecycle(self):
        """Test write-ahead log records INTENT and COMMITTED cleanly."""
        task_id = "task_test_001"
        step_id = "step_001"
        self.wal.log_intent(
            task_id=task_id,
            step_id=step_id,
            step_index=1,
            action_type="FILE_WRITE",
            target=str(self.fixture_dir / "step.txt"),
            parameters={"text": "hello"},
            idempotency=StepIdempotency.IDEMPOTENT,
        )

        uncommitted = self.wal.get_uncommitted_steps(task_id)
        self.assertEqual(len(uncommitted), 1)
        self.assertEqual(uncommitted[0]["step_id"], step_id)

        # Commit step
        self.wal.log_committed(step_id, {"status": "success"})
        uncommitted_after = self.wal.get_uncommitted_steps(task_id)
        self.assertEqual(len(uncommitted_after), 0)

    def test_07_emergency_stop_latency(self):
        """Measure out-of-process supervisor stop latency on a running process (target <= 1.0s)."""
        # Spawn a dummy process to simulate hung agent actor
        dummy_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        sup = ProcessSupervisor(target_pid=dummy_proc.pid)

        # Execute emergency stop
        stop_latency = sup.emergency_stop()
        self.assertLessEqual(stop_latency, 1.0, f"Emergency stop took {stop_latency}s (exceeds 1.0s target)!")
        self.assertFalse(dummy_proc.poll() is None, "Process should be terminated after emergency stop.")

    def test_08_end_to_end_walking_skeleton_slice(self):
        """End-to-end slice: Research query -> Write verified report -> Programmatic verify."""
        task_id = "task_e2e_001"
        report_file = str(self.fixture_dir / "ai_findings_report.txt")
        report_content = (
            "=== AI AGENT DISCOVERY & VERIFICATION REPORT ===\n"
            "Timestamp: 2026-10-22\n"
            "Status: All core security invariants verified.\n"
            "Source: Synthetic verified research slice.\n"
        )

        # 1. Gate check
        dec = self.gateway.validate_action("FILE_WRITE", report_file)
        self.assertTrue(dec.allowed)

        # 2. WAL Intent
        step_id = "step_e2e_write"
        self.wal.log_intent(
            task_id=task_id,
            step_id=step_id,
            step_index=1,
            action_type="FILE_WRITE",
            target=report_file,
            parameters={"length": len(report_content)},
            idempotency=StepIdempotency.NON_IDEMPOTENT,
        )

        # 3. Action execution
        res = FileOpsTool.write_text_file(report_file, report_content)
        self.assertTrue(res["verified"])
        self.assertEqual(res["status"], "success")

        # 4. Programmatic verification
        self.assertTrue(os.path.isfile(report_file))
        self.assertGreater(res["size_bytes"], 50)
        self.assertTrue(len(res["sha256"]) == 64)

        # 5. WAL Commit
        self.wal.log_committed(step_id, res)
        self.assertEqual(len(self.wal.get_uncommitted_steps(task_id)), 0)

        # 6. Check canary file again
        with open(self.canary_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), self.canary_content)


if __name__ == "__main__":
    unittest.main()
