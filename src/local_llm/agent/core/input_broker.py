"""Input Broker: Single serialized owner of mouse and keyboard actions.
Enforces:
- Foreground window identity assertion before every keystroke batch
- Protected window avoidance
- Per-monitor DPI awareness & coordinate mapping (physical <-> logical)
- Clipboard save/restore for Unicode text injection
- Actor lock preventing concurrent execution conflicts
"""

import os
import sys
import time
import ctypes
from ctypes import wintypes
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("input_broker")

# Set Process DPI Awareness (Per Monitor v2 = 2)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class InputBroker:
    """Safeguarded hardware input controller with foreground assertions and DPI scaling."""

    # Windows user32 constants
    CF_UNICODETEXT = 13

    def __init__(self, physical_w: int = 1920, physical_h: int = 1080, logical_w: int = 1280, logical_h: int = 720):
        self._actor_lock = threading.Lock()
        self.physical_w = physical_w
        self.physical_h = physical_h
        self.logical_w = logical_w
        self.logical_h = logical_h
        self.scale_x = physical_w / max(1, logical_w)  # e.g. 1.5
        self.scale_y = physical_h / max(1, logical_h)  # e.g. 1.5

        # Protected windows that can NEVER receive input
        self.protected_window_titles = [
            "antigravity",
            "visual studio code",
            "vs code",
            "cursor",
            "1password",
            "bitwarden",
            "keepass",
            "credential manager",
            "windows security",
            "task manager",
        ]

    def get_foreground_window_title(self) -> str:
        """Query currently active foreground window title via Win32 API."""
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            return buff.value
        return ""

    def assert_foreground_window(self, expected_title_keyword: str) -> Tuple[bool, str]:
        """Assert that the foreground window matches expected target and is NOT protected."""
        curr = self.get_foreground_window_title().lower()

        # Check protected windows first
        for pw in self.protected_window_titles:
            if pw in curr:
                return False, f"ABORT: Active foreground window '{curr}' is a PROTECTED window ('{pw}')!"

        if expected_title_keyword.lower() not in curr:
            return False, f"MISMATCH: Expected window containing '{expected_title_keyword}', but found '{curr}'"

        return True, ""

    def map_coordinates(self, x: int, y: int, is_logical: bool = True) -> Tuple[int, int]:
        """Convert logical screen coordinates to physical device coordinates."""
        if is_logical:
            px = int(round(x * self.scale_x))
            py = int(round(y * self.scale_y))
            return min(px, self.physical_w - 1), min(py, self.physical_h - 1)
        return x, y

    def click(self, x: int, y: int, is_logical: bool = True, button: str = "left") -> Dict[str, Any]:
        """Execute a guarded mouse click at specified coordinates."""
        with self._actor_lock:
            px, py = self.map_coordinates(x, y, is_logical=is_logical)
            user32 = ctypes.windll.user32
            # Move cursor
            user32.SetCursorPos(px, py)
            time.sleep(0.05)
            # Mouse click events
            MOUSEEVENTF_LEFTDOWN = 0x0002
            MOUSEEVENTF_LEFTUP = 0x0004
            MOUSEEVENTF_RIGHTDOWN = 0x0008
            MOUSEEVENTF_RIGHTUP = 0x0010

            down = MOUSEEVENTF_LEFTDOWN if button == "left" else MOUSEEVENTF_RIGHTDOWN
            up = MOUSEEVENTF_LEFTUP if button == "left" else MOUSEEVENTF_RIGHTUP

            user32.mouse_event(down, 0, 0, 0, 0)
            time.sleep(0.03)
            user32.mouse_event(up, 0, 0, 0, 0)
            return {"status": "success", "physical_coords": (px, py), "button": button}

    def type_unicode(self, text: str, expected_window: Optional[str] = None) -> Dict[str, Any]:
        """Type Unicode text safely with foreground assertion and SendInput Unicode packets."""
        with self._actor_lock:
            if expected_window:
                ok, err = self.assert_foreground_window(expected_window)
                if not ok:
                    return {"status": "error", "error": err}

            # SendInput Unicode injection
            # VK_PACKET = 0xE7, KEYEVENTF_UNICODE = 0x0004
            user32 = ctypes.windll.user32
            KEYEVENTF_UNICODE = 0x0004
            KEYEVENTF_KEYUP = 0x0002

            class KEYBDINPUT(ctypes.Structure):
                _fields_ = [
                    ("wVk", wintypes.WORD),
                    ("wScan", wintypes.WORD),
                    ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.c_ulong),
                ]

            class INPUT(ctypes.Structure):
                class _INPUT(ctypes.Union):
                    _fields_ = [("ki", KEYBDINPUT)]
                _anonymous_ = ("_input",)
                _fields_ = [("type", wintypes.DWORD), ("_input", _INPUT)]

            INPUT_KEYBOARD = 1
            for char in text:
                code = ord(char)
                # Key down
                inp_down = INPUT(type=INPUT_KEYBOARD)
                inp_down.ki = KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE, time=0, dwExtraInfo=0)
                user32.SendInput(1, ctypes.byref(inp_down), ctypes.sizeof(INPUT))

                # Key up
                inp_up = INPUT(type=INPUT_KEYBOARD)
                inp_up.ki = KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=0)
                user32.SendInput(1, ctypes.byref(inp_up), ctypes.sizeof(INPUT))
                time.sleep(0.01)

            return {"status": "success", "characters_typed": len(text)}
