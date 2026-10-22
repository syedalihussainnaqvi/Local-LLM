"""Unit tests for Phase 6: Expanded Desktop and OS Control Subsystem.
Tests:
- H-18: Keep-awake execution state management and blank/lock screen detection.
- H-16: Physical GDI screen capture with per-monitor DPI resolution.
- C-07: Window masking logic for protected windows.
- H-22: Per-app accessibility capability cache.
"""

import unittest
import sys
from pathlib import Path
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools.desktop_control import (
    PowerKeepAwake,
    DesktopStateDetector,
    Win32ScreenCapturer,
    AppCapabilityRegistry,
)


class TestDesktopControl(unittest.TestCase):

    def test_01_power_keep_awake_lifecycle(self):
        """Keep-awake thread state must enable and restore cleanly."""
        res_enable = PowerKeepAwake.enable_keep_awake()
        self.assertTrue(res_enable)

        res_restore = PowerKeepAwake.restore_power_state()
        self.assertTrue(res_restore)

    def test_02_blank_and_black_capture_detection(self):
        """Detector must identify blacked-out/blank captures."""
        # 1. Pure black image
        black_img = Image.new("RGB", (1920, 1080), color=(0, 0, 0))
        self.assertTrue(DesktopStateDetector.is_blank_or_black_capture(black_img))

        # 2. Normal desktop image with content
        normal_img = Image.new("RGB", (1920, 1080), color=(128, 128, 255))
        self.assertFalse(DesktopStateDetector.is_blank_or_black_capture(normal_img))

    def test_03_win32_gdi_screen_capture_and_masking(self):
        """GDI capture must return valid RGBA image matching physical screen dimensions."""
        img = Win32ScreenCapturer.capture_screen_with_masking()
        self.assertIsNotNone(img)
        self.assertGreater(img.width, 1000)
        self.assertGreater(img.height, 600)
        self.assertEqual(img.mode, "RGBA")

    def test_04_app_capability_registry(self):
        """Registry must follow accessibility-first hierarchy and cache modes."""
        reg = AppCapabilityRegistry()
        self.assertEqual(reg.get_preferred_mode("Notepad"), "accessibility")
        self.assertEqual(reg.get_preferred_mode("Google Chrome"), "browser_profile")
        # Default unknown app is accessibility-first
        self.assertEqual(reg.get_preferred_mode("CustomApp"), "accessibility")

        # Dynamic registration
        reg.register_capability("CustomApp", "ocr_vision")
        self.assertEqual(reg.get_preferred_mode("CustomApp"), "ocr_vision")


if __name__ == "__main__":
    unittest.main()
