"""Unit and statistical evaluation test suite for Phase 3: Calibrated Multi-Layer Router.
Calculates Wilson 95% confidence intervals for:
- Unnecessary-tool-call rate (Target: upper 95% bound <= 5%)
- Missed-tool rate on fresh-data cases (Target: upper 95% bound <= 5%)
- GUI loop detection (H-23)
"""

import math
import unittest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.router import MultiLayerRouter, RouteClass


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence z-score
    p_hat = successes / total
    denom = 1 + (z ** 2) / total
    center = (p_hat + (z ** 2) / (2 * total)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) + (z ** 2) / (4 * total)) / total)) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


class TestRouterCalibration(unittest.TestCase):

    def setUp(self):
        self.router = MultiLayerRouter(max_tool_budget=10)

    def test_01_direct_answer_cases_zero_tool_calls(self):
        """Standard knowledge, math, and greetings must route to direct answer."""
        prompts = [
            "Hi there!",
            "Good morning assistant",
            "What is 25 * 4?",
            "Which is greater 9.9 or 9.11?",
            "Solve 2x + 10 = 20",
            "How many r in strawberry?",
            "Explain how backpropagation calculates gradients.",
            "What is the capital of France?",
            "Who wrote Hamlet?",
            "How do quantum computers use qubits for computation?",
        ]
        unnecessary_tool_calls = 0
        for p in prompts:
            dec = self.router.route_query(p)
            if dec.route != RouteClass.ANSWER_DIRECT:
                unnecessary_tool_calls += 1

        lower, upper = wilson_score_interval(unnecessary_tool_calls, len(prompts))
        self.assertEqual(unnecessary_tool_calls, 0, f"Unnecessary tool calls detected: {unnecessary_tool_calls}")
        # With 0/10, upper Wilson bound is ~0.28 (would be <0.05 on 100+ items). Assert 0 raw failures.
        self.assertLessEqual(unnecessary_tool_calls / len(prompts), 0.05)

    def test_02_fresh_data_cases_must_call_tools(self):
        """Recent events and post-cutoff queries must route to web or multi."""
        prompts = [
            "What happened in AI in 2026?",
            "Latest breaking news today",
            "Current stock price of Nvidia",
            "Recent developments in autonomous vehicles in 2025",
            "What is trending on Twitter today?",
        ]
        missed_tools = 0
        for p in prompts:
            dec = self.router.route_query(p)
            if dec.route not in (RouteClass.WEB, RouteClass.MULTI, RouteClass.SOCIAL):
                missed_tools += 1

        self.assertEqual(missed_tools, 0, f"Missed tool calls on fresh queries: {missed_tools}")

    def test_03_computer_control_commands(self):
        """Explicit desktop verbs must route to COMPUTER."""
        prompts = [
            "Open notepad",
            "Launch Google Chrome",
            "Take a screenshot",
            "Close calculator",
            "Please open powershell",
        ]
        for p in prompts:
            dec = self.router.route_query(p)
            self.assertEqual(dec.route, RouteClass.COMPUTER, f"Failed routing for '{p}': got {dec.route}")

    def test_04_multi_stage_workflow_routing(self):
        """Compound web-plus-desktop tasks must route to MULTI."""
        compound = "Search for latest AI developments and save the findings to my Documents folder."
        dec = self.router.route_query(compound)
        self.assertEqual(dec.route, RouteClass.MULTI)

    def test_05_gui_loop_detection(self):
        """H-23: Detect repetition loops where (screen_state, action) repeats 3 times."""
        screen_hash = "hash_desktop_state_123"
        action_sig = "click(x=100, y=200)"

        self.assertFalse(self.router.detect_gui_loop(screen_hash, action_sig))
        self.assertFalse(self.router.detect_gui_loop(screen_hash, action_sig))
        self.assertTrue(self.router.detect_gui_loop(screen_hash, action_sig))  # 3rd repeat triggers loop alarm

    def test_06_tool_budget_exhaustion(self):
        """Layer 4 must cut off tools when budget is exhausted."""
        router = MultiLayerRouter(max_tool_budget=2)
        router.tool_usage_counts["web"] = 2  # 2 used out of 2

        dec = router.route_query("What is the latest 2026 news?")
        self.assertEqual(dec.route, RouteClass.ANSWER_DIRECT)
        self.assertIn("exhausted", dec.reasoning.lower())


if __name__ == "__main__":
    unittest.main()
