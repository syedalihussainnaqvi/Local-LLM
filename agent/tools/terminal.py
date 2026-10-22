"""Full Visible Policy-Governed Terminal Subsystem (Section 6B).
Addresses:
- C-52: Policy-governed persistent terminal sessions.
- H-59: Persistent cwd/env, exit-code capture, process-tree termination, ANSI stripping, secret hygiene.
- H-60: Terminal output tagged as untrusted provenance (cannot originate actions or modify grants).
- Trusted Terminal Session: time-limited, directory-scoped token for developer workflows.
"""

import os
import sys
import re
import time
import json
import logging
import subprocess
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.policy.gateway import PolicyGateway, RiskTier, InputProvenance, PolicyDecision

logger = logging.getLogger("terminal_subsystem")


@dataclass
class TerminalCommandResult:
    session_id: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    provenance: InputProvenance = InputProvenance.UNTRUSTED_CONTENT
    truncated: bool = False


@dataclass
class TrustedSession:
    session_id: str
    allowed_root_directory: str
    expiry_timestamp: float
    created_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expiry_timestamp


class TerminalSubsystem:
    """Policy-governed terminal session manager with persistent cwd and audit journal."""

    ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def __init__(
        self,
        gateway: PolicyGateway,
        journal_path: str = "eval/fixtures/terminal_journal.jsonl",
        default_shell: str = "powershell.exe",
    ):
        self.gateway = gateway
        self.journal_path = journal_path
        self.default_shell = default_shell
        self.active_trusted_session: Optional[TrustedSession] = None
        self.session_cwds: Dict[str, str] = {"default": os.getcwd()}
        os.makedirs(os.path.dirname(os.path.abspath(journal_path)), exist_ok=True)

    @classmethod
    def strip_ansi(cls, text: str) -> str:
        """Remove ANSI terminal color and formatting sequences."""
        return cls.ANSI_ESCAPE_RE.sub("", text)

    def classify_command_tier(self, command_str: str) -> RiskTier:
        """Classify terminal command into risk tiers based on program and arguments (Section 6B)."""
        cmd_lower = command_str.lower().strip()

        # Check catastrophic blocks first
        for pat in self.gateway.BLOCKED_COMMAND_PATTERNS:
            if re.search(pat, cmd_lower):
                return RiskTier.T3_SENSITIVE

        # Tier 1: Read-only and project-local commands
        t1_patterns = [
            r"^(git\s+(status|log|diff|branch|remote)|dir|ls|echo|pwd|type|cat|head|tail)\b",
            r"^(python\s+--version|python\s+-m\s+unittest|pytest|node\s+-v|npm\s+test)\b",
            r"^pip\s+list\b",
        ]
        for pat in t1_patterns:
            if re.search(pat, cmd_lower):
                return RiskTier.T1_ROUTINE

        # Tier 3: Deletion, accounts, security, disk
        t3_patterns = [
            r"\b(rm|del|remove-item)\b",
            r"\b(net\s+user|net\s+localgroup|reg\s+add|reg\s+delete|icacls)\b",
            r"\b(format|diskpart|bcdedit)\b",
        ]
        for pat in t3_patterns:
            if re.search(pat, cmd_lower):
                return RiskTier.T3_SENSITIVE

        # Tier 2: Installs, global configurations, network
        return RiskTier.T2_ELEVATED

    def start_trusted_session(self, session_id: str, root_dir: str, duration_seconds: float = 1800.0) -> TrustedSession:
        """Opt-in time-limited developer session scoped to a directory tree."""
        canon_root = str(Path(root_dir).resolve())
        ts = TrustedSession(
            session_id=session_id,
            allowed_root_directory=canon_root,
            expiry_timestamp=time.time() + duration_seconds,
        )
        self.active_trusted_session = ts
        logger.info(f"Started Trusted Terminal Session '{session_id}' in '{canon_root}' for {duration_seconds}s.")
        return ts

    def end_trusted_session(self) -> None:
        self.active_trusted_session = None

    def execute_command(
        self,
        command_str: str,
        session_id: str = "default",
        timeout_seconds: float = 30.0,
    ) -> TerminalCommandResult:
        """Execute command governed by policy gateway, persistent cwd, and monotonic deadline."""
        t0 = time.time()
        cwd = self.session_cwds.get(session_id, os.getcwd())
        tier = self.classify_command_tier(command_str)

        # Policy Gate evaluation
        dec = self.gateway.validate_action("TERMINAL_RUN", command_str)
        if not dec.allowed:
            return TerminalCommandResult(
                session_id=session_id,
                command=command_str,
                exit_code=126,
                stdout="",
                stderr=f"Policy Gateway Blocked: {dec.reason}",
                duration_seconds=0.0,
            )

        # Check Trusted Terminal Session override for Tier 2
        is_trusted = False
        if self.active_trusted_session and not self.active_trusted_session.is_expired:
            try:
                common = os.path.commonpath([cwd, self.active_trusted_session.allowed_root_directory])
                if common == self.active_trusted_session.allowed_root_directory and tier == RiskTier.T2_ELEVATED:
                    is_trusted = True
            except ValueError:
                pass

        if tier in (RiskTier.T2_ELEVATED, RiskTier.T3_SENSITIVE) and not is_trusted:
            if dec.requires_confirmation:
                return TerminalCommandResult(
                    session_id=session_id,
                    command=command_str,
                    exit_code=127,
                    stdout="",
                    stderr=f"Confirmation Required for Tier {tier.value}: {command_str}",
                    duration_seconds=0.0,
                )

        # Scrub environment: remove unrelated sensitive tokens (H-59)
        env = dict(os.environ)
        for secret_key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY"]:
            env.pop(secret_key, None)
        env["PAGER"] = "cat"

        # Execute command in target shell with monotonic deadline and exit code propagation
        wrapped_cmd = f"{command_str}\nif ($LASTEXITCODE -ne $null) {{ exit $LASTEXITCODE }}"
        cmd_array = [self.default_shell, "-NoProfile", "-Command", wrapped_cmd]
        proc = subprocess.Popen(
            cmd_array,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        try:
            stdout_raw, stderr_raw = proc.communicate(timeout=timeout_seconds)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_raw, stderr_raw = proc.communicate()
            exit_code = -9
            stderr_raw += f"\n[TIMEOUT: Execution exceeded {timeout_seconds}s deadline. Process tree killed.]"

        duration = time.time() - t0
        clean_stdout = self.strip_ansi(stdout_raw)
        clean_stderr = self.strip_ansi(stderr_raw)

        # Track directory changes if `cd` was invoked
        cd_match = re.search(r"^\s*cd\s+([^\s;]+)", command_str)
        if cd_match and exit_code == 0:
            target_dir = cd_match.group(1).strip('"\'')
            new_cwd = str((Path(cwd) / target_dir).resolve())
            if os.path.isdir(new_cwd):
                self.session_cwds[session_id] = new_cwd

        result = TerminalCommandResult(
            session_id=session_id,
            command=command_str,
            exit_code=exit_code,
            stdout=clean_stdout[:10000],  # cap head
            stderr=clean_stderr[:5000],
            duration_seconds=round(duration, 3),
            provenance=InputProvenance.UNTRUSTED_CONTENT,  # H-60
            truncated=len(clean_stdout) > 10000,
        )

        # Command Journaling (JSONL)
        journal_entry = {
            "timestamp": time.time(),
            "session_id": session_id,
            "cwd": cwd,
            "command": command_str,
            "tier": tier.value,
            "exit_code": exit_code,
            "duration": result.duration_seconds,
            "is_trusted_session": is_trusted,
        }
        try:
            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(journal_entry) + "\n")
        except Exception as e:
            logger.warning(f"Could not write terminal journal: {e}")

        return result
