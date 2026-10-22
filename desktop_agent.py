"""Autonomous Desktop & OS Agent Module with Deep Planning Engine.
Enables full autonomous computer control with multi-stage task decomposition:
  - Deep Task Planner: breaks down complex, compound, multi-app user goals into ordered phases & atomic steps
  - Application lifecycle management (open/close WhatsApp, Chrome, VS Code, Terminal, Notepad, etc.)
  - In-app workflows:
      * WhatsApp: search contact, send message, delete message
      * Chrome: search Google, scroll pages, open ChatGPT and submit tasks
  - Terminal & PowerShell command generation and execution
  - File system operations (create, read, edit, delete files)
  - Mouse & Keyboard automation (clicks, moves, scrolling, typing via clipboard, hotkeys, screenshots)
  - Mandatory User Permission & Safety Guard System with interactive approval and conversational confirmation
"""

import os
import sys
import time
import json
import logging
import re
import uuid
import subprocess
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("desktop_agent")

# Try importing GUI automation libraries with graceful fallbacks
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.15
    HAS_PYAUTOGUI = True
except Exception as e:
    logger.warning(f"pyautogui import failed: {e}")
    HAS_PYAUTOGUI = False

try:
    import pyperclip
    HAS_PYPERCLIP = True
except Exception:
    HAS_PYPERCLIP = False

try:
    import pygetwindow as gw
    HAS_PYGETWINDOW = True
except Exception:
    HAS_PYGETWINDOW = False


# ── Action Risk Levels ───────────────────────────────────────────────────────
class RiskLevel:
    SAFE = "SAFE"          # Reading files, system info, taking screenshots
    LOW = "LOW"            # Opening URLs, opening standard apps, scrolling
    MEDIUM = "MEDIUM"      # Typing text, mouse clicks, closing apps, creating files
    HIGH = "HIGH"          # Running terminal commands, sending WhatsApp messages, editing files
    CRITICAL = "CRITICAL"  # Deleting files, killing critical system processes


@dataclass
class ActionStep:
    step_id: str
    phase: str             # e.g., 'Phase 1: WhatsApp Automation', 'Phase 2: Web Research', etc.
    action_type: str       # e.g., 'OPEN_APP', 'WHATSAPP_SEND', 'RUN_TERMINAL', 'FILE_DELETE', etc.
    description: str
    parameters: Dict[str, Any]
    risk_level: str
    status: str = "PENDING_APPROVAL"  # PENDING_APPROVAL, APPROVED, REJECTED, EXECUTED, FAILED
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class ActionPlan:
    plan_id: str
    user_goal: str
    phases: List[str]
    steps: List[ActionStep]
    risk_summary: str
    plan_summary: str
    status: str = "PENDING_PERMISSION"  # PENDING_PERMISSION, APPROVED, REJECTED, EXECUTING, COMPLETED, PARTIALLY_FAILED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    executed_at: Optional[str] = None


# ── Robust Desktop Executor ─────────────────────────────────────────────────
class DesktopExecutor:
    """Executes validated OS and desktop automation actions with maximum reliability."""

    # Common Windows application mappings
    APP_COMMANDS = {
        "whatsapp": ["cmd", "/c", "start", "whatsapp:"],
        "chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"
            if os.path.isfile(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
            else "chrome"
        ],
        "google chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"
            if os.path.isfile(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
            else "chrome"
        ],
        "notepad": ["notepad.exe"],
        "calc": ["calc.exe"],
        "calculator": ["calc.exe"],
        "cmd": ["cmd.exe"],
        "terminal": ["powershell.exe"],
        "explorer": ["explorer.exe"],
        "file explorer": ["explorer.exe"],
        "file exprorer": ["explorer.exe"],
        "microsoft store": ["cmd", "/c", "start", "ms-windows-store:"],
        "ms-windows-store": ["cmd", "/c", "start", "ms-windows-store:"],
        "store": ["cmd", "/c", "start", "ms-windows-store:"],
        "code": ["code"],
        "vscode": ["code"],
        "edge": ["cmd", "/c", "start", "msedge"],
        "spotify": ["cmd", "/c", "start", "spotify:"],
    }

    @staticmethod
    def get_chrome_executable() -> str:
        """Find reliable Google Chrome binary path on Windows."""
        standard_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ]
        for p in standard_paths:
            if os.path.isfile(p):
                return p
        return "chrome"

    @staticmethod
    def open_app(app_name: str, path: Optional[str] = None, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Open or launch an application safely."""
        clean_name = app_name.lower().strip()
        if clean_name in ("chrome", "google chrome"):
            chrome_bin = DesktopExecutor.get_chrome_executable()
            cmd = [chrome_bin]
        elif path:
            cmd = [path]
        else:
            cmd = list(DesktopExecutor.APP_COMMANDS.get(clean_name, ["cmd", "/c", "start", clean_name]))

        if args:
            cmd.extend(args)

        try:
            # If command starts with chrome binary or direct exe, spawn without shell
            proc = subprocess.Popen(cmd, shell=(cmd[0] == "cmd"))
            time.sleep(1.2)
            return {"status": "success", "message": f"Launched '{app_name}' (PID: {proc.pid})", "pid": proc.pid}
        except Exception as e:
            # Fallback to shell start
            try:
                subprocess.Popen(f"start {clean_name}", shell=True)
                time.sleep(1.2)
                return {"status": "success", "message": f"Launched '{app_name}' via Windows shell fallback"}
            except Exception as e2:
                return {"status": "error", "error": f"Failed to launch '{app_name}': {e2}"}

    @staticmethod
    def close_app(app_name: str) -> Dict[str, Any]:
        """Close an application cleanly or terminate if hung."""
        clean_name = app_name.lower().strip()
        proc_name = clean_name if clean_name.endswith(".exe") else f"{clean_name}.exe"

        # Alias mappings
        aliases = {
            "google chrome.exe": "chrome.exe",
            "google chrome": "chrome.exe",
            "whatsapp desktop.exe": "WhatsApp.exe",
            "whatsapp.exe": "WhatsApp.exe",
            "terminal.exe": "powershell.exe",
            "vs code.exe": "Code.exe",
            "vscode.exe": "Code.exe",
        }
        proc_name = aliases.get(proc_name, proc_name)

        try:
            res = subprocess.run(["taskkill", "/IM", proc_name, "/F"], capture_output=True, text=True)
            if res.returncode == 0:
                return {"status": "success", "message": f"Closed '{app_name}' (process: {proc_name})"}
            else:
                return {"status": "warning", "message": f"Process '{proc_name}' close command executed: {res.stdout or res.stderr}"}
        except Exception as e:
            return {"status": "error", "error": f"Failed to close '{app_name}': {e}"}

    @staticmethod
    def focus_window(title_keyword: str) -> bool:
        """Find and focus a window by title keyword with retries."""
        if not HAS_PYGETWINDOW:
            return False
        try:
            keyword_lower = title_keyword.lower()
            all_wins = gw.getAllWindows()
            matching = [w for w in all_wins if keyword_lower in (w.title or "").lower()]
            if matching:
                win = matching[0]
                if win.isMinimized:
                    win.restore()
                win.activate()
                time.sleep(0.4)
                return True
        except Exception as e:
            logger.debug(f"Focus window error for '{title_keyword}': {e}")
        return False

    @staticmethod
    def mouse_click(x: Optional[int] = None, y: Optional[int] = None, button: str = "left", clicks: int = 1) -> Dict[str, Any]:
        """Click mouse at coordinates or current position."""
        if not HAS_PYAUTOGUI:
            return {"status": "error", "error": "pyautogui is not available"}
        try:
            if x is not None and y is not None:
                pyautogui.click(x=x, y=y, button=button, clicks=clicks)
            else:
                pyautogui.click(button=button, clicks=clicks)
            return {"status": "success", "message": f"Clicked {button} (x={x}, y={y}, clicks={clicks})"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def mouse_move(x: int, y: int, duration: float = 0.8) -> Dict[str, Any]:
        """Move mouse cursor smoothly to coordinates."""
        if not HAS_PYAUTOGUI:
            return {"status": "error", "error": "pyautogui is not available"}
        try:
            pyautogui.moveTo(x, y, duration=duration)
            return {"status": "success", "message": f"Moved mouse cursor to ({x}, {y}) in {duration}s"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def mouse_scroll(clicks: int) -> Dict[str, Any]:
        """Scroll mouse wheel (positive = up, negative = down)."""
        if not HAS_PYAUTOGUI:
            return {"status": "error", "error": "pyautogui is not available"}
        try:
            pyautogui.scroll(clicks)
            return {"status": "success", "message": f"Scrolled mouse wheel ({clicks} clicks)"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def keyboard_type(text: str, enter: bool = False, live_type: bool = True) -> Dict[str, Any]:
        """Type text character-by-character so user watches it type live on the screen, with clipboard paste fallback."""
        if not HAS_PYAUTOGUI:
            return {"status": "error", "error": "pyautogui is not available"}
        try:
            is_ascii = all(ord(c) < 128 for c in text)
            if is_ascii and live_type:
                pyautogui.write(text, interval=0.05)
            elif HAS_PYPERCLIP:
                pyperclip.copy(text)
                time.sleep(0.05)
                pyautogui.hotkey("ctrl", "v")
            else:
                pyautogui.write(text, interval=0.02)

            if enter:
                time.sleep(0.3)
                pyautogui.press("enter")
            return {"status": "success", "message": f"Typed text live on screen: '{text}' ({len(text)} characters)"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def keyboard_press(key: str) -> Dict[str, Any]:
        """Press a single key (e.g., 'enter', 'esc', 'tab', 'backspace')."""
        if not HAS_PYAUTOGUI:
            return {"status": "error", "error": "pyautogui is not available"}
        try:
            pyautogui.press(key)
            return {"status": "success", "message": f"Pressed key '{key}'"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def keyboard_hotkey(*keys: str) -> Dict[str, Any]:
        """Press hotkey combination (e.g. ['ctrl', 'a'], ['alt', 'tab'])."""
        if not HAS_PYAUTOGUI:
            return {"status": "error", "error": "pyautogui is not available"}
        try:
            pyautogui.hotkey(*keys)
            return {"status": "success", "message": f"Pressed hotkey: {' + '.join(keys)}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def capture_screenshot(save_path: Optional[str] = None) -> Dict[str, Any]:
        """Capture screen image and save to disk."""
        if not HAS_PYAUTOGUI:
            return {"status": "error", "error": "pyautogui is not available"}
        try:
            out_path = save_path or os.path.join(os.getcwd(), f"screenshot_{int(time.time())}.png")
            img = pyautogui.screenshot()
            img.save(out_path)
            return {"status": "success", "file": out_path, "size": list(img.size), "message": f"Screenshot saved to {out_path}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── High-Level In-App Workflows ──────────────────────────────────────────
    @staticmethod
    def whatsapp_send_message(recipient: str, message: str) -> Dict[str, Any]:
        """Open WhatsApp, search contact/chat, and send message."""
        # 1. Open WhatsApp Desktop or fallback to Web
        open_res = DesktopExecutor.open_app("whatsapp")
        time.sleep(2.5)

        # 2. Focus window
        focused = DesktopExecutor.focus_window("WhatsApp")
        if not focused:
            # Fallback to WhatsApp Web in Chrome
            web_url = f"https://web.whatsapp.com/send?text={urllib.parse.quote(message)}"
            DesktopExecutor.open_app("chrome", args=[web_url])
            time.sleep(3.0)
            DesktopExecutor.focus_window("WhatsApp")

        if HAS_PYAUTOGUI:
            # 3. Focus search bar via Ctrl+F
            pyautogui.hotkey("ctrl", "f")
            time.sleep(0.4)

            # 4. Type recipient contact name and press Enter
            DesktopExecutor.keyboard_type(recipient, enter=True)
            time.sleep(1.5)

            # 5. Type and send the message
            DesktopExecutor.keyboard_type(message, enter=True)
            time.sleep(0.5)

            return {
                "status": "success",
                "recipient": recipient,
                "message": f"Sent message to '{recipient}': '{message[:50]}...'",
            }
        else:
            return {
                "status": "warning",
                "recipient": recipient,
                "message": f"Launched WhatsApp for recipient '{recipient}', but pyautogui is required for automated typing",
            }

    @staticmethod
    def whatsapp_delete_message() -> Dict[str, Any]:
        """Delete the most recent message in the active WhatsApp conversation."""
        DesktopExecutor.focus_window("WhatsApp")
        time.sleep(0.5)
        if HAS_PYAUTOGUI:
            pyautogui.press("esc")
            time.sleep(0.2)
            # Context menu shortcut on Windows
            pyautogui.hotkey("shift", "f10")
            time.sleep(0.4)
            DesktopExecutor.keyboard_type("delete", enter=True)
            time.sleep(0.4)
            pyautogui.press("enter")
            return {"status": "success", "message": "Initiated message deletion in WhatsApp"}
        return {"status": "error", "error": "pyautogui required for automated keyboard input"}

    @staticmethod
    def browser_search(query: str) -> Dict[str, Any]:
        """Open Google Chrome and search for query."""
        search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"
        res = DesktopExecutor.open_app("chrome", args=[search_url])
        time.sleep(2.0)
        DesktopExecutor.focus_window("Google")
        return {"status": "success", "query": query, "url": search_url, "message": f"Opened Google Search for '{query}'"}

    @staticmethod
    def browser_open_url(url: str) -> Dict[str, Any]:
        """Open specific URL in Google Chrome."""
        res = DesktopExecutor.open_app("chrome", args=[url])
        time.sleep(2.0)
        return {"status": "success", "url": url, "message": f"Opened URL '{url}' in Google Chrome"}

    @staticmethod
    def browser_scroll(direction: str = "down", amount: int = 600) -> Dict[str, Any]:
        """Scroll active browser window."""
        DesktopExecutor.focus_window("Chrome")
        time.sleep(0.3)
        clicks = -amount if direction == "down" else amount
        return DesktopExecutor.mouse_scroll(clicks)

    @staticmethod
    def browser_chatgpt_prompt(task_prompt: str) -> Dict[str, Any]:
        """Open ChatGPT in Google Chrome and submit task prompt."""
        chatgpt_url = "https://chatgpt.com/"
        DesktopExecutor.open_app("chrome", args=[chatgpt_url])
        time.sleep(3.5)
        DesktopExecutor.focus_window("ChatGPT")
        time.sleep(0.6)

        if HAS_PYAUTOGUI:
            # Type task prompt with enter to submit
            DesktopExecutor.keyboard_type(task_prompt, enter=True)
            return {
                "status": "success",
                "message": f"Opened ChatGPT and submitted prompt: '{task_prompt[:60]}...'",
                "prompt": task_prompt,
            }
        return {"status": "warning", "message": "Opened ChatGPT, but automated typing requires GUI focus"}

    @staticmethod
    def run_terminal_command(command: str, cwd: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
        """Execute a command in PowerShell with safety timeout and capture full output."""
        try:
            start_t = time.time()
            res = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
                cwd=cwd or os.getcwd(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = round((time.time() - start_t) * 1000, 2)
            out = res.stdout.strip()
            err = res.stderr.strip()
            return {
                "status": "success" if res.returncode == 0 else "error",
                "command": command,
                "stdout": out,
                "stderr": err,
                "exit_code": res.returncode,
                "duration_ms": duration,
                "message": f"Command executed in {duration}ms (exit code: {res.returncode})",
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "command": command, "error": f"Command timed out after {timeout} seconds"}
        except Exception as e:
            return {"status": "error", "command": command, "error": str(e)}

    @staticmethod
    def create_file(file_path: str, content: str = "") -> Dict[str, Any]:
        """Create a file with full directory path support."""
        try:
            abs_path = os.path.abspath(file_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "success", "file": abs_path, "bytes": len(content), "message": f"Created file '{abs_path}' ({len(content)} bytes)"}
        except Exception as e:
            return {"status": "error", "error": f"Failed to create file: {e}"}

    @staticmethod
    def edit_file(file_path: str, content: str, mode: str = "overwrite") -> Dict[str, Any]:
        """Edit or append to an existing file."""
        try:
            abs_path = os.path.abspath(file_path)
            write_mode = "a" if mode == "append" else "w"
            with open(abs_path, write_mode, encoding="utf-8") as f:
                f.write(content)
            return {"status": "success", "file": abs_path, "mode": mode, "bytes_written": len(content), "message": f"Updated file '{abs_path}'"}
        except Exception as e:
            return {"status": "error", "error": f"Failed to edit file: {e}"}

    @staticmethod
    def delete_file(file_path: str) -> Dict[str, Any]:
        """Delete a file safely."""
        try:
            abs_path = os.path.abspath(file_path)
            if os.path.isfile(abs_path):
                os.remove(abs_path)
                return {"status": "success", "deleted": abs_path, "message": f"Deleted file '{abs_path}'"}
            elif os.path.isdir(abs_path):
                os.rmdir(abs_path)
                return {"status": "success", "deleted_dir": abs_path, "message": f"Deleted directory '{abs_path}'"}
            return {"status": "error", "error": f"File or directory not found: '{abs_path}'"}
        except Exception as e:
            return {"status": "error", "error": f"Failed to delete '{file_path}': {e}"}

    @staticmethod
    def read_file(file_path: str, max_lines: int = 100) -> Dict[str, Any]:
        """Read content from a file."""
        try:
            abs_path = os.path.abspath(file_path)
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [f.readline() for _ in range(max_lines)]
            content = "".join(lines)
            return {"status": "success", "file": abs_path, "content": content, "lines": len(lines)}
        except Exception as e:
            return {"status": "error", "error": f"Failed to read file '{file_path}': {e}"}

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Probe system status, screen resolution, and active windows."""
        info = {
            "platform": sys.platform,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "screen": list(pyautogui.size()) if HAS_PYAUTOGUI else "Unavailable",
            "active_windows": [],
        }
        if HAS_PYGETWINDOW:
            try:
                info["active_windows"] = [w.title for w in gw.getAllWindows() if w.title][:10]
            except Exception:
                pass
        return info


# ── Deep Autonomous Task Planner ─────────────────────────────────────────────
class DeepDesktopPlanner:
    """
    Decomposes complex, composite, multi-step user prompts into structured,
    ordered, risk-assessed ActionPlans with logical phases and atomic steps.
    """

    @staticmethod
    def plan_from_prompt(prompt: str) -> ActionPlan:
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        p_lower = prompt.lower().strip()
        steps: List[ActionStep] = []
        phases: List[str] = []
        step_idx = 1

        # Check for composite capabilities in user prompt:
        has_whatsapp = any(w in p_lower for w in ["whatsapp", "whatspp", "whatapp"])
        has_chatgpt = "chatgpt" in p_lower
        has_google = "google" in p_lower or "search" in p_lower or "chrome" in p_lower
        has_scroll = "scroll" in p_lower
        has_terminal = any(w in p_lower for w in ["terminal", "powershell", "cmd", "command", "generate command"])
        has_create_file = "create file" in p_lower or "make file" in p_lower or "create delete edit file" in p_lower
        has_edit_file = "edit file" in p_lower or "modify file" in p_lower or "create delete edit file" in p_lower
        has_delete_file = "delete file" in p_lower or "remove file" in p_lower or "create delete edit file" in p_lower
        has_close_app = "close app" in p_lower or "close apps" in p_lower
        has_open_app = "open app" in p_lower or "open apps" in p_lower or "open " in p_lower
        has_mouse_key = any(w in p_lower for w in ["mouse", "keyboard", "full charge", "laptop mouse keybord"])
        has_store = any(w in p_lower for w in ["microsoft store", "windows store", "ms store"]) or ("store" in p_lower and "microsoft" in p_lower)
        has_explorer = any(w in p_lower for w in ["file explorer", "file exprorer", "explorer"])
        has_mouse_moving = any(w in p_lower for w in ["mouse moving", "mouse move", "move mouse", "live mouse"])
        has_live_typing = any(w in p_lower for w in ["typing", "type text", "live typing"])

        # ── Phase: Windows Core Applications ────────────────────────────────
        if has_explorer or has_store:
            phases.append("Phase: Windows Core Applications")
            if has_explorer:
                steps.append(ActionStep(
                    step_id=f"{plan_id}_{step_idx}",
                    phase="Phase: Windows Core Applications",
                    action_type="OPEN_APP",
                    description="Launch Windows File Explorer",
                    parameters={"app_name": "file explorer"},
                    risk_level=RiskLevel.LOW,
                ))
                step_idx += 1

            if has_store:
                steps.append(ActionStep(
                    step_id=f"{plan_id}_{step_idx}",
                    phase="Phase: Windows Core Applications",
                    action_type="OPEN_APP",
                    description="Launch Microsoft Store Application",
                    parameters={"app_name": "microsoft store"},
                    risk_level=RiskLevel.LOW,
                ))
                step_idx += 1

        # ── Phase 1: WhatsApp Workflows ──────────────────────────────────────
        if has_whatsapp:
            phases.append("Phase: WhatsApp Automation")
            # 1. Open WhatsApp
            steps.append(ActionStep(
                step_id=f"{plan_id}_{step_idx}",
                phase="Phase: WhatsApp Automation",
                action_type="OPEN_APP",
                description="Launch WhatsApp Desktop Application",
                parameters={"app_name": "whatsapp"},
                risk_level=RiskLevel.LOW,
            ))
            step_idx += 1

            # Extract recipient and message cleanly
            recipient = "Contact"
            msg = "Hello! Task delegated by Autonomous Agent."
            send_match = re.search(r"send\s+(?:a\s+)?message\s+(?:saying\s+|with\s+text\s+)?['\"]([^'\"]+)['\"](?:\s+to\s+(['\"a-zA-Z0-9\s\+\-]+))?", p_lower)
            if send_match:
                msg = send_match.group(1).strip()
                if send_match.group(2) and send_match.group(2).strip() not in ("him", "them", "me"):
                    recipient = send_match.group(2).strip()
            else:
                simple_to = re.search(r"to\s+([A-Za-z0-9\+\-]+)(?:\s+saying|\s+with|\s*$)", p_lower)
                if simple_to and simple_to.group(1).strip() not in ("him", "them", "me", "do", "the"):
                    recipient = simple_to.group(1).strip()

            steps.append(ActionStep(
                step_id=f"{plan_id}_{step_idx}",
                phase="Phase: WhatsApp Automation",
                action_type="WHATSAPP_SEND",
                description=f"Send WhatsApp message to '{recipient}': '{msg}'",
                parameters={"recipient": recipient, "message": msg},
                risk_level=RiskLevel.HIGH,
            ))
            step_idx += 1

            if "delete message" in p_lower:
                steps.append(ActionStep(
                    step_id=f"{plan_id}_{step_idx}",
                    phase="Phase: WhatsApp Automation",
                    action_type="WHATSAPP_DELETE",
                    description="Delete the sent message in WhatsApp",
                    parameters={},
                    risk_level=RiskLevel.HIGH,
                ))
                step_idx += 1

        # ── Phase 2: Google Chrome & Web Browsing ────────────────────────────
        if has_google:
            phases.append("Phase: Google Search & Web Exploration")
            # Priority 1: Quoted search term (most explicit)
            search_match = re.search(r"(?:open\s+google\s+search|search\s+google|search)\s+(?:for\s+)?['\"]([^'\"]+)['\"]", p_lower)
            if search_match:
                query = search_match.group(1).strip()
            else:
                # Priority 2: "search for X" or "search X" unquoted (stop at common delimiters)
                search_for_match = re.search(
                    r"(?:search\s+for|google\s+search\s+for|search)\s+([a-zA-Z0-9][^,;.\n]+?)(?:\s+(?:and|then|also|scroll|open|on)\b|$)",
                    prompt, re.IGNORECASE
                )
                if search_for_match:
                    raw_q = search_for_match.group(1).strip()
                    query = raw_q if raw_q else "latest news"
                else:
                    # Priority 3: "X on google" pattern
                    on_google_match = re.search(r"(.+?)\s+on\s+google", p_lower)
                    if on_google_match:
                        query = on_google_match.group(1).strip()
                    else:
                        query = "latest artificial intelligence developments"

            steps.append(ActionStep(
                step_id=f"{plan_id}_{step_idx}",
                phase="Phase: Google Search & Web Exploration",
                action_type="BROWSER_SEARCH",
                description=f"Open Google Chrome and search for: '{query}'",
                parameters={"query": query},
                risk_level=RiskLevel.LOW,
            ))
            step_idx += 1

            if has_scroll:
                steps.append(ActionStep(
                    step_id=f"{plan_id}_{step_idx}",
                    phase="Phase: Google Search & Web Exploration",
                    action_type="BROWSER_SCROLL",
                    description="Scroll down the search results page smoothly (600px)",
                    parameters={"direction": "down", "amount": 600},
                    risk_level=RiskLevel.LOW,
                ))
                step_idx += 1

        # ── Phase: Mouse & Keyboard Live Demonstration ──────────────────────
        if has_mouse_moving or has_live_typing:
            phases.append("Phase: Mouse & Keyboard Live Demonstration")
            if has_mouse_moving:
                steps.append(ActionStep(
                    step_id=f"{plan_id}_{step_idx}",
                    phase="Phase: Mouse & Keyboard Live Demonstration",
                    action_type="MOUSE_MOVE",
                    description="Demonstrate live smooth mouse cursor navigation across the screen (x=680, y=420)",
                    parameters={"x": 680, "y": 420, "duration": 1.2},
                    risk_level=RiskLevel.LOW,
                ))
                step_idx += 1

            if has_live_typing:
                type_text = query if (has_google and query) else "Autonomous Agent live typing demonstration"
                steps.append(ActionStep(
                    step_id=f"{plan_id}_{step_idx}",
                    phase="Phase: Mouse & Keyboard Live Demonstration",
                    action_type="KEYBOARD_TYPE",
                    description=f"Demonstrate live character-by-character typing into active window: '{type_text}'",
                    parameters={"text": type_text, "enter": True, "live_type": True},
                    risk_level=RiskLevel.MEDIUM,
                ))
                step_idx += 1

        # ── Phase 3: ChatGPT on Google Chrome ────────────────────────────────
        if has_chatgpt:
            phases.append("Phase: ChatGPT AI Delegation")
            task_match = re.search(r"(?:give\s+task|prompt|ask\s+chatgpt)\s+(?:to\s+him\s+)?['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
            if task_match:
                chatgpt_task = task_match.group(1).strip()
            else:
                chatgpt_task = "Analyze the latest AI research breakthroughs and outline practical autonomous agent implementations."

            steps.append(ActionStep(
                step_id=f"{plan_id}_{step_idx}",
                phase="Phase: ChatGPT AI Delegation",
                action_type="CHATGPT_PROMPT",
                description=f"Open ChatGPT on Google Chrome and submit task: '{chatgpt_task[:60]}...'",
                parameters={"task_prompt": chatgpt_task},
                risk_level=RiskLevel.MEDIUM,
            ))
            step_idx += 1

        # ── Phase 4: Terminal & PowerShell Command Execution ─────────────────
        if has_terminal:
            phases.append("Phase: Terminal & System Commands")
            # Extract explicit command if provided in quotes or after colon
            explicit_cmd = re.search(r"(?:command|run|execute)\s*[:\s-]\s*['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
            if explicit_cmd:
                cmd = explicit_cmd.group(1).strip()
            else:
                # Generate a real, high-value, safe PowerShell diagnostic command
                cmd = "Get-Date; Write-Output '=== System Health & Diagnostics ==='; Get-Process | Select-Object -First 5 -Property ProcessName, CPU, WorkingSet"

            steps.append(ActionStep(
                step_id=f"{plan_id}_{step_idx}",
                phase="Phase: Terminal & System Commands",
                action_type="RUN_TERMINAL",
                description=f"Execute PowerShell command: '{cmd}'",
                parameters={"command": cmd},
                risk_level=RiskLevel.HIGH,
            ))
            step_idx += 1

        # ── Phase 5: File Operations (Create, Edit, Delete) ───────────────────
        if has_create_file or has_edit_file or has_delete_file:
            phases.append("Phase: File System Operations")
            if has_create_file:
                steps.append(ActionStep(
                    step_id=f"{plan_id}_{step_idx}",
                    phase="Phase: File System Operations",
                    action_type="CREATE_FILE",
                    description="Create task summary file 'agent_workspace\\notes.txt'",
                    parameters={"file_path": "agent_workspace\\notes.txt", "content": f"Task Plan Generated by Autonomous Desktop Agent at {datetime.now().isoformat()}\nGoal: {prompt}\n"},
                    risk_level=RiskLevel.MEDIUM,
                ))
                step_idx += 1

            if has_edit_file:
                steps.append(ActionStep(
                    step_id=f"{plan_id}_{step_idx}",
                    phase="Phase: File System Operations",
                    action_type="EDIT_FILE",
                    description="Append execution status log to 'agent_workspace\\notes.txt'",
                    parameters={"file_path": "agent_workspace\\notes.txt", "content": "\n[STATUS: PLAN AUTHORIZED AND EXECUTING]\n", "mode": "append"},
                    risk_level=RiskLevel.MEDIUM,
                ))
                step_idx += 1

            if has_delete_file:
                steps.append(ActionStep(
                    step_id=f"{plan_id}_{step_idx}",
                    phase="Phase: File System Operations",
                    action_type="DELETE_FILE",
                    description="Safely remove temporary scratch file 'agent_workspace\\temp.tmp'",
                    parameters={"file_path": "agent_workspace\\temp.tmp"},
                    risk_level=RiskLevel.HIGH,
                ))
                step_idx += 1

        # ── Phase 6: App Lifecycle (Close Apps) ──────────────────────────────
        if has_close_app and not (has_whatsapp or has_google):
            phases.append("Phase: Application Management")
            app_match = re.search(r"close\s+(?:apps?|application)\s*(['\"a-zA-Z0-9_\-]*)", p_lower)
            target = app_match.group(1).strip() if (app_match and app_match.group(1).strip()) else "notepad"
            steps.append(ActionStep(
                step_id=f"{plan_id}_{step_idx}",
                phase="Phase: Application Management",
                action_type="CLOSE_APP",
                description=f"Terminate application: '{target}'",
                parameters={"app_name": target},
                risk_level=RiskLevel.MEDIUM,
            ))
            step_idx += 1

        # ── Fallback / General Delegation Plan ────────────────────────────────
        if not steps:
            phases.append("Phase: System Readiness & Diagnostics")
            steps.append(ActionStep(
                step_id=f"{plan_id}_1",
                phase="Phase: System Readiness & Diagnostics",
                action_type="SYSTEM_INFO",
                description="Query system state, active windows, and screen resolution",
                parameters={},
                risk_level=RiskLevel.SAFE,
            ))
            steps.append(ActionStep(
                step_id=f"{plan_id}_2",
                phase="Phase: System Readiness & Diagnostics",
                action_type="RUN_TERMINAL",
                description="Check system readiness and active processes in PowerShell",
                parameters={"command": "Get-Date; Get-ComputerInfo | Select-Object CsName, OsName; Get-Process | Select-Object -First 5"},
                risk_level=RiskLevel.HIGH,
            ))

        # Risk Calculation
        risk_priority = [RiskLevel.SAFE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        highest_risk = max((s.risk_level for s in steps), key=lambda r: risk_priority.index(r), default=RiskLevel.LOW)

        plan_summary = (
            f"Autonomous Desktop Execution Plan contains {len(steps)} steps across {len(phases)} phases.\n"
            f"Highest Risk: {highest_risk}. Explicit user permission is strictly required before execution."
        )

        return ActionPlan(
            plan_id=plan_id,
            user_goal=prompt,
            phases=phases,
            steps=steps,
            risk_summary=highest_risk,
            plan_summary=plan_summary,
            status="PENDING_PERMISSION",
        )


# ── Desktop Agent Singleton & API ───────────────────────────────────────────
class DesktopAgent:
    """Central agent orchestrating plans, permission authorization, and execution."""

    def __init__(self):
        self.executor = DesktopExecutor()
        self.planner = DeepDesktopPlanner()
        self.active_plans: Dict[str, ActionPlan] = {}
        self.latest_pending_plan_id: Optional[str] = None

    def plan_task(self, prompt: str) -> ActionPlan:
        """Create a deep planned sequence of steps requiring user permission."""
        plan = self.planner.plan_from_prompt(prompt)
        self.active_plans[plan.plan_id] = plan
        self.latest_pending_plan_id = plan.plan_id
        return plan

    def get_plan(self, plan_id: str) -> Optional[ActionPlan]:
        return self.active_plans.get(plan_id)

    def get_pending_plan(self) -> Optional[ActionPlan]:
        """Retrieve the latest plan awaiting permission."""
        if self.latest_pending_plan_id:
            plan = self.active_plans.get(self.latest_pending_plan_id)
            if plan and plan.status == "PENDING_PERMISSION":
                return plan
        # Scan for any pending plan
        for plan in reversed(list(self.active_plans.values())):
            if plan.status == "PENDING_PERMISSION":
                return plan
        return None

    @staticmethod
    def is_approval_intent(prompt: str) -> bool:
        """Check if user prompt is granting permission to execute a pending plan."""
        p = prompt.lower().strip()
        approval_keywords = [
            "approve", "proceed", "yes", "grant", "grant permission", "execute",
            "do it", "go ahead", "sure", "ok", "okay", "confirm", "start",
            "i approve", "permission granted", "run the plan", "run it"
        ]
        return any(p == kw or p.startswith(kw + " ") or p.endswith(" " + kw) for kw in approval_keywords)

    @staticmethod
    def is_rejection_intent(prompt: str) -> bool:
        """Check if user prompt is rejecting permission."""
        p = prompt.lower().strip()
        rejection_keywords = [
            "reject", "cancel", "no", "deny", "stop", "abort", "don't", "do not"
        ]
        return any(p == kw or p.startswith(kw + " ") for kw in rejection_keywords)

    def grant_permission(self, plan_id: str) -> Dict[str, Any]:
        """User explicitly grants permission to execute plan."""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return {"status": "error", "error": f"Plan '{plan_id}' not found"}
        plan.status = "APPROVED"
        for s in plan.steps:
            s.status = "APPROVED"
        return {"status": "success", "message": f"Permission GRANTED for plan {plan_id}", "plan": asdict(plan)}

    def reject_permission(self, plan_id: str) -> Dict[str, Any]:
        """User explicitly rejects permission."""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return {"status": "error", "error": f"Plan '{plan_id}' not found"}
        plan.status = "REJECTED"
        for s in plan.steps:
            s.status = "REJECTED"
        if self.latest_pending_plan_id == plan_id:
            self.latest_pending_plan_id = None
        return {"status": "success", "message": f"Permission REJECTED for plan {plan_id}"}

    def prompt_windows_native_popup(self, plan_id: str) -> bool:
        """
        Pops up a native top-most Windows modal dialog box with [Yes] (Allow) and [No] (Decline).
        Returns True if user allowed, False if declined.
        """
        plan = self.active_plans.get(plan_id)
        if not plan:
            return False

        if sys.platform == "win32":
            try:
                import ctypes
                MB_YESNO = 0x4
                MB_ICONEXCLAMATION = 0x30
                MB_TOPMOST = 0x40000
                MB_SETFOREGROUND = 0x10000
                IDYES = 6

                step_lines = "\n".join([f"  • [{s.phase}] {s.description} ({s.risk_level})" for s in plan.steps[:6]])
                if len(plan.steps) > 6:
                    step_lines += f"\n  ... and {len(plan.steps) - 6} more step(s)"

                msg = (
                    f"⚠️ AUTONOMOUS DESKTOP PERMISSION REQUEST\n\n"
                    f"Plan ID: {plan.plan_id}\n"
                    f"Goal: {plan.user_goal}\n"
                    f"Risk Level: {plan.risk_summary}\n\n"
                    f"Requested Actions:\n{step_lines}\n\n"
                    f"Do you want to ALLOW the agent to control your mouse, keyboard, applications, and terminal?\n\n"
                    f"Click [Yes] to ALLOW & EXECUTE, or [No] to DECLINE."
                )
                title = f"Autonomous Desktop Agent - Permission Required ({plan.plan_id})"
                res = ctypes.windll.user32.MessageBoxW(0, msg, title, MB_YESNO | MB_ICONEXCLAMATION | MB_TOPMOST | MB_SETFOREGROUND)
                allowed = (res == IDYES)
                if allowed:
                    self.grant_permission(plan_id)
                else:
                    self.reject_permission(plan_id)
                return allowed
            except Exception as e:
                logger.warning(f"Native Windows popup error: {e}")
        return False

    def execute_plan(self, plan_id: str, auto_approve: bool = False) -> Generator[Dict[str, Any], None, None]:
        """Execute action plan step by step with streaming progress and strict safety verification."""
        plan = self.active_plans.get(plan_id)
        if not plan:
            yield {"type": "error", "message": f"Plan '{plan_id}' not found"}
            return

        if plan.status != "APPROVED" and not auto_approve:
            yield {
                "type": "permission_required",
                "plan_id": plan.plan_id,
                "message": f"Action Plan {plan_id} requires explicit user permission before execution.",
                "steps": [asdict(s) for s in plan.steps],
            }
            return

        plan.status = "EXECUTING"
        plan.executed_at = datetime.now(timezone.utc).isoformat()
        yield {"type": "status", "action": "starting", "message": f"Executing authorized plan ({len(plan.steps)} steps)..."}

        for idx, step in enumerate(plan.steps, 1):
            yield {
                "type": "status",
                "action": "step_start",
                "step_id": step.step_id,
                "phase": step.phase,
                "description": step.description,
                "message": f"[{step.phase}] Step {idx}/{len(plan.steps)}: {step.description} [{step.risk_level}]",
            }

            res = None
            try:
                t = step.action_type
                p = step.parameters

                if t == "OPEN_APP":
                    res = self.executor.open_app(p.get("app_name", ""), p.get("path"), p.get("args"))
                elif t == "CLOSE_APP":
                    res = self.executor.close_app(p.get("app_name", ""))
                elif t == "WHATSAPP_SEND":
                    res = self.executor.whatsapp_send_message(p.get("recipient", "Contact"), p.get("message", "Hello!"))
                elif t == "WHATSAPP_DELETE":
                    res = self.executor.whatsapp_delete_message()
                elif t == "BROWSER_SEARCH":
                    res = self.executor.browser_search(p.get("query", ""))
                elif t == "BROWSER_SCROLL":
                    res = self.executor.browser_scroll(p.get("direction", "down"), p.get("amount", 600))
                elif t == "BROWSER_CHATGPT" or t == "CHATGPT_PROMPT":
                    res = self.executor.browser_chatgpt_prompt(p.get("task_prompt", ""))
                elif t == "BROWSER_OPEN_URL":
                    res = self.executor.browser_open_url(p.get("url", ""))
                elif t == "RUN_TERMINAL":
                    res = self.executor.run_terminal_command(p.get("command", ""))
                elif t == "CREATE_FILE":
                    res = self.executor.create_file(p.get("file_path", ""), p.get("content", ""))
                elif t == "EDIT_FILE":
                    res = self.executor.edit_file(p.get("file_path", ""), p.get("content", ""), p.get("mode", "overwrite"))
                elif t == "DELETE_FILE":
                    res = self.executor.delete_file(p.get("file_path", ""))
                elif t == "READ_FILE":
                    res = self.executor.read_file(p.get("file_path", ""), p.get("max_lines", 100))
                elif t == "MOUSE_CLICK":
                    res = self.executor.mouse_click(p.get("x"), p.get("y"), p.get("button", "left"), p.get("clicks", 1))
                elif t == "MOUSE_MOVE":
                    res = self.executor.mouse_move(p.get("x", 500), p.get("y", 500), duration=p.get("duration", 0.8))
                elif t == "MOUSE_SCROLL":
                    res = self.executor.mouse_scroll(p.get("clicks", -300))
                elif t == "KEYBOARD_TYPE":
                    res = self.executor.keyboard_type(p.get("text", ""), enter=p.get("enter", False), live_type=p.get("live_type", True))
                elif t == "KEYBOARD_PRESS":
                    res = self.executor.keyboard_press(p.get("key", "enter"))
                elif t == "KEYBOARD_HOTKEY":
                    res = self.executor.keyboard_hotkey(*p.get("keys", ["ctrl", "c"]))
                elif t == "SCREENSHOT":
                    res = self.executor.capture_screenshot(p.get("save_path"))
                elif t == "SYSTEM_INFO":
                    res = self.executor.get_system_info()
                else:
                    res = {"status": "error", "error": f"Unknown action type: {t}"}

                step.result = res
                step.status = "EXECUTED" if res.get("status") in ("success", "warning") else "FAILED"

            except Exception as e:
                step.status = "FAILED"
                step.error = str(e)
                res = {"status": "error", "error": str(e)}

            yield {
                "type": "status",
                "action": "step_finish",
                "step_id": step.step_id,
                "status": step.status,
                "result": res,
                "message": f"Step {idx} completed: {step.description} ({step.status})",
            }

        all_ok = all(s.status == "EXECUTED" for s in plan.steps)
        plan.status = "COMPLETED" if all_ok else "PARTIALLY_FAILED"
        if self.latest_pending_plan_id == plan.plan_id:
            self.latest_pending_plan_id = None

        yield {
            "type": "plan_complete",
            "plan_id": plan.plan_id,
            "status": plan.status,
            "plan": asdict(plan),
            "message": f"Plan execution finished with status: {plan.status}.",
        }


# ── Global Singleton ────────────────────────────────────────────────────────
_GLOBAL_DESKTOP_AGENT: Optional[DesktopAgent] = None


def get_desktop_agent() -> DesktopAgent:
    global _GLOBAL_DESKTOP_AGENT
    if _GLOBAL_DESKTOP_AGENT is None:
        _GLOBAL_DESKTOP_AGENT = DesktopAgent()
    return _GLOBAL_DESKTOP_AGENT
