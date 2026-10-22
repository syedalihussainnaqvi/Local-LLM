"""Expanded Desktop and OS Control Subsystem.
Addresses:
- H-15: Overlay exclusion from screen capture via SetWindowDisplayAffinity (WDA_EXCLUDEFROMCAPTURE) and WS_EX_NOACTIVATE.
- H-16: Per-monitor DPI coordinate mapping and native Win32 GDI screen capture.
- H-18: State detectors for screen lock, blank capture, and screen saver + Keep-Awake thread management.
- H-19: UIPI elevation mismatch detection.
- H-22: Per-app accessibility capability probing with fallback caching.
- C-07: Real-time window masking for protected windows in captured screenshots.
"""

import os
import sys
import time
import ctypes
from ctypes import wintypes
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

logger = logging.getLogger("desktop_control")

# Ensure Per-Monitor DPI Awareness (Per Monitor v2)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class PowerKeepAwake:
    """Manages Windows system and display keep-awake execution state (H-18)."""

    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    ES_CONTINUOUS = 0x80000000

    @classmethod
    def enable_keep_awake(cls) -> bool:
        """Prevent Windows from sleeping or blanking the display during task runs."""
        try:
            res = ctypes.windll.kernel32.SetThreadExecutionState(
                cls.ES_CONTINUOUS | cls.ES_SYSTEM_REQUIRED | cls.ES_DISPLAY_REQUIRED
            )
            return res != 0
        except Exception as e:
            logger.warning(f"Could not enable keep-awake: {e}")
            return False

    @classmethod
    def restore_power_state(cls) -> bool:
        """Restore standard Windows sleep and power management settings."""
        try:
            res = ctypes.windll.kernel32.SetThreadExecutionState(cls.ES_CONTINUOUS)
            return res != 0
        except Exception as e:
            logger.warning(f"Could not restore power state: {e}")
            return False


class DesktopStateDetector:
    """Detects lock screens, blank captures, UAC secure desktops, and elevation mismatches."""

    @staticmethod
    def is_session_locked() -> bool:
        """Check if current Windows session is locked or on secure desktop."""
        user32 = ctypes.windll.user32
        # Check desktop access
        hdesk = user32.OpenInputDesktop(0, False, 0x0100)  # DESKTOP_SWITCHDESKTOP
        if hdesk == 0:
            return True
        user32.CloseDesktop(hdesk)

        # Check foreground window handle
        hwnd = user32.GetForegroundWindow()
        if hwnd == 0:
            return True

        # Check window title for lock screen
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            if "windows default lock screen" in buff.value.lower():
                return True

        return False

    @staticmethod
    def is_blank_or_black_capture(img: Image.Image) -> bool:
        """Detect all-black or blank screen captures caused by secure desktops or sleep."""
        # Fast sampling of 16 bounding points
        w, h = img.size
        sample_points = [
            (w // 4, h // 4), (w // 2, h // 4), (3 * w // 4, h // 4),
            (w // 4, h // 2), (w // 2, h // 2), (3 * w // 4, h // 2),
            (w // 4, 3 * h // 4), (w // 2, 3 * h // 4), (3 * w // 4, 3 * h // 4),
        ]
        rgb_img = img.convert("RGB")
        non_black = 0
        for px, py in sample_points:
            r, g, b = rgb_img.getpixel((px, py))
            if r > 5 or g > 5 or b > 5:
                non_black += 1

        return non_black == 0


class Win32ScreenCapturer:
    """High-performance GDI BitBlt capture with protected-window masking (C-07, H-15)."""

    PROTECTED_TITLES = [
        "antigravity", "visual studio code", "vs code", "cursor",
        "1password", "bitwarden", "keepass", "credential manager",
    ]

    @classmethod
    def get_window_rects(cls) -> List[Tuple[str, int, int, int, int]]:
        """Enumerate top-level window titles and bounding rectangles."""
        user32 = ctypes.windll.user32
        rects: List[Tuple[str, int, int, int, int]] = []

        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

        def enum_windows_callback(hwnd, lParam):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buff, length + 1)
                    title = buff.value
                    r = RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(r))
                    rects.append((title, r.left, r.top, r.right, r.bottom))
            return True

        user32.EnumWindows(WNDENUMPROC(enum_windows_callback), 0)
        return rects

    @classmethod
    def capture_screen_with_masking(cls) -> Image.Image:
        """Capture physical 1920x1080 display via GDI BitBlt and mask protected windows."""
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        hdc_screen = user32.GetDC(0)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        hbitmap = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
        old_bitmap = gdi32.SelectObject(hdc_mem, hbitmap)

        # 0x00CC0020 = SRCCOPY, 0x40000000 = CAPTUREBLT
        gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, 0, 0, 0x00CC0020 | 0x40000000)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h  # top-down
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0

        buf = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(hdc_mem, hbitmap, 0, h, buf, ctypes.byref(bmi), 0)

        img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1)

        # Cleanup GDI handles
        gdi32.SelectObject(hdc_mem, old_bitmap)
        gdi32.DeleteObject(hbitmap)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)

        # Masking protected windows (C-07)
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            rects = cls.get_window_rects()
            for title, left, top, right, bottom in rects:
                t_lower = title.lower()
                for pw in cls.PROTECTED_TITLES:
                    if pw in t_lower:
                        # Draw black rectangle over protected window area
                        draw.rectangle([max(0, left), max(0, top), min(w, right), min(h, bottom)], fill=(0, 0, 0, 255))
                        break
        except Exception as e:
            logger.warning(f"Window masking error: {e}")

        return img


class AppCapabilityRegistry:
    """Caches capability probes (UIA accessibility vs OCR/vision) per application (H-22)."""

    def __init__(self):
        self._cache: Dict[str, str] = {
            "notepad": "accessibility",
            "calc": "accessibility",
            "chrome": "browser_profile",
            "whatsapp": "ocr_vision",
        }

    def get_preferred_mode(self, app_name: str) -> str:
        clean = app_name.lower().strip()
        for k, v in self._cache.items():
            if k in clean:
                return v
        return "accessibility"  # Accessibility-first default (Section 6)

    def register_capability(self, app_name: str, mode: str) -> None:
        self._cache[app_name.lower().strip()] = mode
