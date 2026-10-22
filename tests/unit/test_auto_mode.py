"""Unit tests for Phase 9: Auto Mode Bounded Autonomy Controller.
Tests:
- Section 8: Hard safety limits (max_steps, wall clock, consecutive failures).
- Section 8: Stall detection and one-time replan escalation.
- Section 8: Progress update tracking.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.policy.gateway import PolicyGateway
from agent.core.wal import WALManager
from agent.core.execution_engine import ExecutionEngine
from agent.core.auto_mode import AutoController, AutoLimits, AutoStopReason


class TestAutoMode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = PROJECT_ROOT / "eval" / "fixtures"
        cls.fixture_dir.mkdir(parents=True, exist_ok=True)
        cls.wal_db = cls.fixture_dir / "auto_test_wal.db"
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

    def test_01_normal_bounded_task_completion(self):
        """Auto mode completes standard workflow within limits."""
        controller = AutoController(self.engine, limits=AutoLimits(max_steps=5, max_wall_clock_seconds=30.0))
        res = controller.run_bounded_task(
            task_id="auto_task_001",
            prompt="Research and summarize AI ethics",
            target_filename="auto_ethics_report.txt",
            report_content="AI Ethics Verified Report Content\n",
        )
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["stop_reason"], AutoStopReason.GOAL_VERIFIED.value)
        self.assertGreater(res["progress_updates"], 0)

    def test_02_max_steps_hard_limit(self):
        """Auto mode stops immediately when max_steps is reached."""
        controller = AutoController(self.engine, limits=AutoLimits(max_steps=1))
        # Plan has 2 steps; limit is 1 step
        res = controller.run_bounded_task(
            task_id="auto_task_step_limit",
            prompt="Research and save",
            target_filename="step_limit.txt",
            report_content="Payload",
        )
        self.assertEqual(res["status"], "stopped")
        self.assertEqual(res["stop_reason"], AutoStopReason.MAX_STEPS_REACHED.value)

    def test_03_wall_clock_timeout_limit(self):
        """Auto mode halts when wall clock exceeds timeout."""
        controller = AutoController(self.engine, limits=AutoLimits(max_wall_clock_seconds=0.0001))
        res = controller.run_bounded_task(
            task_id="auto_task_timeout",
            prompt="Research and save",
            target_filename="timeout.txt",
            report_content="Payload",
        )
        self.assertEqual(res["status"], "stopped")
        self.assertEqual(res["stop_reason"], AutoStopReason.WALL_CLOCK_TIMEOUT.value)

    def test_04_stall_detector(self):
        """Stall detector detects non-advancing repeated states."""
        controller = AutoController(self.engine, limits=AutoLimits(stall_threshold_steps=3))
        fp = "CLICK_COORDS:(100, 200)"
        self.assertFalse(controller.detect_stall(fp))
        self.assertFalse(controller.detect_stall(fp))
        self.assertTrue(controller.detect_stall(fp))


if __name__ == "__main__":
    unittest.main()
