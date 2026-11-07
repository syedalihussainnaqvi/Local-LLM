"""Browser Tool: Launches visible Google Chrome using a dedicated isolated agent profile.
Never touches the user's personal default browsing profile by default.
"""

import os
import sys
import subprocess
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("browser_tool")


class BrowserTool:
    """Safely launches and controls Google Chrome with a dedicated, isolated user profile."""

    def __init__(self, profile_dir: Optional[str] = None):
        if not profile_dir:
            base_dir = os.path.expandvars(r"%LOCALAPPDATA%\AntigravityAgent\BrowserProfile")
            os.makedirs(base_dir, exist_ok=True)
            self.profile_dir = base_dir
        else:
            self.profile_dir = profile_dir

    @staticmethod
    def find_chrome_binary() -> str:
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return "chrome"

    def launch_visible_search(self, query: str) -> Dict[str, Any]:
        """Launch visible Chrome with dedicated profile pointing to a search query."""
        import urllib.parse
        chrome_bin = self.find_chrome_binary()
        encoded = urllib.parse.quote_plus(query)
        search_url = f"https://www.google.com/search?q={encoded}"

        # Command arguments: dedicated profile, no headless tricks, visible window
        cmd = [
            chrome_bin,
            f"--user-data-dir={self.profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            search_url,
        ]

        try:
            # Spawn with argument array (NO shell=True)
            proc = subprocess.Popen(cmd)
            return {
                "status": "success",
                "pid": proc.pid,
                "url": search_url,
                "profile": self.profile_dir,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
