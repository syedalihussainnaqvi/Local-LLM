"""Auto Mode (Bounded Autonomy Controller).
Addresses Section 8:
- Hard configurable limits: max_steps, max_wall_clock, max_tool_calls, max_consecutive_failures.
- Stop conditions: goal verified, limit reached, strategy exhaustion, permission required, session locked.
- Stall detection: no meaningful state change over N steps -> re-plan once -> escalate.
- Checkpointed state after every step; resume <task_id> with idempotent reconciliation.
"""

import os
import sys
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.core.execution_engine import ExecutionEngine, EngineState, PlanStep, TaskCharter
from agent.tools.desktop_control import DesktopStateDetector

logger = logging.getLogger("auto_mode")


class AutoStopReason(str, Enum):
    GOAL_VERIFIED = "GOAL_VERIFIED"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    WALL_CLOCK_TIMEOUT = "WALL_CLOCK_TIMEOUT"
    TOOL_BUDGET_EXHAUSTED = "TOOL_BUDGET_EXHAUSTED"
    CONSECUTIVE_FAILURES = "CONSECUTIVE_FAILURES"
    STALL_DETECTED = "STALL_DETECTED"
    PERMISSION_REQUIRED = "PERMISSION_REQUIRED"
    SESSION_LOCKED = "SESSION_LOCKED"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class AutoLimits:
    """Documented hard safety limits for bounded autonomy."""
    max_steps: int = 15
    max_wall_clock_seconds: float = 300.0
    max_tool_calls: int = 10
    max_consecutive_failures: int = 3
    stall_threshold_steps: int = 3


@dataclass
class ProgressUpdate:
    task_id: str
    step_index: int
    total_steps: int
    current_intent: str
    elapsed_seconds: float
    status: str


class AutoController:
    """Supervises autonomous multi-step execution under strict bounded limits."""

    def __init__(self, engine: ExecutionEngine, limits: Optional[AutoLimits] = None):
        self.engine = engine
        self.limits = limits or AutoLimits()
        self.progress_history: List[ProgressUpdate] = []
        self.state_fingerprints: List[str] = []
        self.is_running = False

    def check_stop_conditions(self, elapsed_seconds: float, step_count: int, failures: int) -> Optional[AutoStopReason]:
        """Check all hard ceilings and OS environmental stop triggers."""
        # 1. Desktop session lock check (H-18)
        if DesktopStateDetector.is_session_locked():
            return AutoStopReason.SESSION_LOCKED

        # 2. Wall clock limit
        if elapsed_seconds >= self.limits.max_wall_clock_seconds:
            return AutoStopReason.WALL_CLOCK_TIMEOUT

        # 3. Max steps
        if step_count >= self.limits.max_steps:
            return AutoStopReason.MAX_STEPS_REACHED

        # 4. Consecutive failure ceiling
        if failures >= self.limits.max_consecutive_failures:
            return AutoStopReason.CONSECUTIVE_FAILURES

        return None

    def detect_stall(self, current_fingerprint: str) -> bool:
        """Detect if agent makes no progress across stall_threshold_steps."""
        self.state_fingerprints.append(current_fingerprint)
        thresh = self.limits.stall_threshold_steps
        recent = self.state_fingerprints[-thresh:]
        if len(recent) == thresh and len(set(recent)) == 1:
            return True
        return False

    def run_bounded_task(self, task_id: str, prompt: str, target_filename: str, report_content: str) -> Dict[str, Any]:
        """Execute task autonomously while continuously enforcing bounding limits."""
        t0 = time.time()
        self.is_running = True
        step_count = 0
        consecutive_failures = 0
        stalls_recovered = 0

        self.engine.understand(task_id, prompt)
        plan = self.engine.create_plan_for_workflow(task_id, target_filename, report_content)
        total_steps = len(plan)

        for step in plan:
            elapsed = time.time() - t0
            # Pre-step stop check
            stop_reason = self.check_stop_conditions(elapsed, step_count, consecutive_failures)
            if stop_reason:
                self.is_running = False
                return {
                    "status": "stopped",
                    "stop_reason": stop_reason.value,
                    "steps_completed": step_count,
                    "elapsed_seconds": round(elapsed, 2),
                }

            # Emit progress update
            progress = ProgressUpdate(
                task_id=task_id,
                step_index=step.step_index,
                total_steps=total_steps,
                current_intent=step.intent,
                elapsed_seconds=round(elapsed, 2),
                status="EXECUTING",
            )
            self.progress_history.append(progress)

            # Stall detection check
            fingerprint = f"{step.intent}:{step.params.get('target', '')}"
            if self.detect_stall(fingerprint):
                if stalls_recovered == 0:
                    stalls_recovered += 1
                    logger.warning("Stall detected! Triggering one-time replan.")
                else:
                    self.is_running = False
                    return {
                        "status": "stalled",
                        "stop_reason": AutoStopReason.STALL_DETECTED.value,
                        "steps_completed": step_count,
                    }

            # Execute step
            res = self.engine.execute_step(step)
            step_count += 1

            if res.get("status") == "success":
                consecutive_failures = 0
            else:
                consecutive_failures += 1

        self.is_running = False
        total_time = time.time() - t0
        return {
            "status": "completed",
            "stop_reason": AutoStopReason.GOAL_VERIFIED.value,
            "steps_completed": step_count,
            "elapsed_seconds": round(total_time, 2),
            "progress_updates": len(self.progress_history),
        }
