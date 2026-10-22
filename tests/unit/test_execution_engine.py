"""Unit tests for Phase 7: Execution Engine, State Machine, Programmatic Verifier,
Failure Ledger, and Idempotent Crash Resume.
Tests:
- C-14: Idempotent resume (never duplicate non-idempotent steps without verification).
- H-34: Programmatic verification first (SHA256, file size, readability).
- Section 7: Task charter pinning and state machine lifecycle.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.policy.gateway import PolicyGateway, RiskTier
from agent.core.wal import WALManager, StepIdempotency, StepState
from agent.core.execution_engine import (
    ExecutionEngine,
    EngineState,
    ProgrammaticVerifier,
    FailureLedger,
)


class TestExecutionEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = PROJECT_ROOT / "eval" / "fixtures"
        cls.fixture_dir.mkdir(parents=True, exist_ok=True)
        cls.wal_db = cls.fixture_dir / "engine_test_wal.db"
        if cls.wal_db.exists():
            cls.wal_db.unlink()

        cls.gateway = PolicyGateway(granted_roots=[str(cls.fixture_dir)])
        cls.wal = WALManager(db_path=str(cls.wal_db))

    def setUp(self):
        self.engine = ExecutionEngine(
            gateway=self.gateway,
            wal=self.wal,
            fixture_dir=str(self.fixture_dir),
        )

    def test_01_programmatic_verifier(self):
        """H-34: Verifier checks file existence, size, and SHA256 checksum."""
        test_file = self.fixture_dir / "verifier_test.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("VERIFICATION_PAYLOAD_TEST")

        ok, meta = ProgrammaticVerifier.verify_file_created(str(test_file), min_size_bytes=5)
        self.assertTrue(ok)
        self.assertEqual(meta["size_bytes"], 25)
        self.assertTrue(len(meta["sha256"]) == 64)

        # Test failure on nonexistent file
        bad_ok, bad_meta = ProgrammaticVerifier.verify_file_created(str(self.fixture_dir / "nonexistent.txt"))
        self.assertFalse(bad_ok)

    def test_02_failure_ledger(self):
        """Failure ledger records failed strategies and detects repetitions."""
        ledger = FailureLedger()
        self.assertFalse(ledger.has_strategy_failed("s1", "coord_click"))

        ledger.record_failure("s1", "coord_click", "Element not clickable at coordinates")
        self.assertTrue(ledger.has_strategy_failed("s1", "coord_click"))
        self.assertFalse(ledger.has_strategy_failed("s1", "keyboard_shortcut"))

    def test_03_full_workflow_execution(self):
        """Engine must complete full research-and-report workflow with verified artifacts."""
        task_id = "task_engine_workflow_001"
        report_file = "ai_autonomous_report.txt"
        report_text = (
            "Autonomous Agent Research Summary\n"
            "Date: 2026-10-22\n"
            "Findings: High-reliability execution state machine operational.\n"
        )

        res = self.engine.run_workflow(
            task_id=task_id,
            prompt="Research autonomous agents and save findings",
            target_filename=report_file,
            report_text=report_text,
        )

        self.assertEqual(res["status"], "completed")
        self.assertEqual(self.engine.current_state, EngineState.COMPLETE)
        self.assertEqual(len(res["verified_artifacts"]), 1)

        # Check physical file exists and content matches
        created_path = Path(res["verified_artifacts"][0])
        self.assertTrue(created_path.exists())
        with open(created_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), report_text)

    def test_04_idempotent_resume_after_simulated_crash(self):
        """C-14: On resume, WAL reconciles uncommitted steps against reality and prevents duplication."""
        task_id = "task_crash_sim_001"
        step_id = "step_crash_file_write"
        target_path = str(self.fixture_dir / "crash_test_target.txt")

        # 1. Simulate file already created before crash
        with open(target_path, "w", encoding="utf-8") as f:
            f.write("DATA_COMMITTED_IN_REALITY_BEFORE_CRASH")

        # 2. Write WAL INTENT without COMMITTED (simulating crash right after write)
        self.wal.log_intent(
            task_id=task_id,
            step_id=step_id,
            step_index=1,
            action_type="FILE_WRITE",
            target=target_path,
            parameters={"target": target_path},
            idempotency=StepIdempotency.NON_IDEMPOTENT,
        )

        # 3. Trigger crash reconciliation
        rec = self.engine.reconcile_on_resume(task_id)
        self.assertEqual(rec["status"], "reconciled")
        actions = rec["actions"]
        self.assertEqual(len(actions), 1)
        # Because file exists in reality, it reconciles as committed rather than duplicating write
        self.assertEqual(actions[0]["action"], "reconciled_committed")

        # 4. Verify uncommitted steps is now zero
        self.assertEqual(len(self.wal.get_uncommitted_steps(task_id)), 0)


if __name__ == "__main__":
    unittest.main()
