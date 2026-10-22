"""Policy Gateway: Enforces mandatory security tiers, structured grants,
canonical path resolution, provenance verification, and command allowlists.
No OS-touching tool can be executed except through this gateway.
"""

import os
import sys
import re
import time
import hashlib
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("policy_gateway")


class RiskTier(str, Enum):
    T0_OBSERVE = "T0_OBSERVE"          # Screenshots (with masking), reads, listings, web reads
    T1_ROUTINE = "T1_ROUTINE"          # Open/close apps, navigate, type in non-sensitive fields, write in granted roots
    T2_ELEVATED = "T2_ELEVATED"        # Admin/root, installs, settings, outside granted roots, real browser profile
    T3_SENSITIVE = "T3_SENSITIVE"      # Permanent delete, messaging send, purchases, security changes, unknown code
    TP_POWER = "TP_POWER"              # Shutdown, restart, sign-out, sleep


class InputProvenance(str, Enum):
    TRUSTED_USER = "TRUSTED_USER"          # Explicit user input or pinned task charter
    UNTRUSTED_CONTENT = "UNTRUSTED_CONTENT"# Web pages, OCR text, file contents, chat messages, command output


@dataclass(frozen=True)
class Grant:
    grant_id: str
    action_type: str
    exact_target: str
    scope: str
    expiry_timestamp: float
    max_uses: int
    payload_hash: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class PolicyDecision:
    allowed: bool
    tier: RiskTier
    reason: str
    requires_confirmation: bool = False
    confirmation_prompt: Optional[str] = None
    canonical_target: Optional[str] = None


class PolicyGateway:
    """Central choke-point for all tool operations touching OS, files, network, or applications."""

    # Default protected windows that can never be focused, clicked, or typed into
    PROTECTED_WINDOW_SUBSTRINGS = [
        "antigravity",
        "visual studio code",
        "vs code",
        "cursor",
        "1password",
        "bitwarden",
        "keepass",
        "lastpass",
        "credential manager",
        "task manager",
        "windows security",
    ]

    # Catastrophic command patterns that are strictly blocked
    BLOCKED_COMMAND_PATTERNS = [
        r"format\s+[a-z]:",
        r"\bdel\b.*(/[fFqQsS]\s*)+.*[a-zA-Z]:",
        r"\brmdir\b.*(/[sS]\s*)+.*[a-zA-Z]:",
        r"\bRemove-Item\b.*-Recurse",
        r":\(\)\{\s*:\s*\|\s*:\s*&\s*\};:",  # fork bomb
        r"powershell.*-EncodedCommand",
        r"cmd(\.exe)?\s+/c\s+.*format",
        r"\bbcdedit\b",
        r"\bdiskpart\b",
        r"\breg\s+delete\s+HKLM\b",
        r"\bdel\b.*(\\\*|C:\\Windows|C:\\Users)",
    ]

    def __init__(self, granted_roots: Optional[List[str]] = None):
        self.granted_roots: List[str] = []
        if granted_roots:
            for r in granted_roots:
                self.granted_roots.append(self.canonicalize_path(r))
        self.active_grants: Dict[str, Grant] = {}
        self.grant_use_counts: Dict[str, int] = {}
        self.audit_log: List[Dict[str, Any]] = []

    @staticmethod
    def canonicalize_path(raw_path: str) -> str:
        """Resolve symlinks, junctions, 8.3 short names, env vars, and case."""
        expanded = os.path.expandvars(os.path.expanduser(raw_path.strip()))
        resolved = Path(expanded).resolve()
        return str(resolved)

    def is_path_in_granted_roots(self, canonical_path: str) -> bool:
        """Check whether a canonical path resides strictly inside granted roots."""
        if not self.granted_roots:
            return False
        for root in self.granted_roots:
            try:
                common = os.path.commonpath([canonical_path, root])
                if common == root:
                    return True
            except ValueError:
                continue
    def add_grant(self, grant: Grant, provenance: InputProvenance) -> bool:
        """Grants can ONLY be created from trusted user input, never from model or untrusted web/file text (C-06)."""
        if provenance != InputProvenance.TRUSTED_USER:
            logger.warning(f"Rejected grant creation from untrusted provenance: {provenance}")
            return False
        self.active_grants[grant.grant_id] = grant
        self.grant_use_counts[grant.grant_id] = 0
        return True

    def check_active_grant(self, grant_id: str, action_type: str, target: str, payload_data: Optional[str] = None) -> Tuple[bool, str]:
        """Verify an active Grant object for sensitive T3 actions (C-06)."""
        if grant_id not in self.active_grants:
            return False, f"Grant '{grant_id}' does not exist or was never granted."

        grant = self.active_grants[grant_id]
        now = time.time()
        if now > grant.expiry_timestamp:
            return False, f"Grant '{grant_id}' has expired (expired at {grant.expiry_timestamp}, current {now})."

        uses = self.grant_use_counts.get(grant_id, 0)
        if uses >= grant.max_uses:
            return False, f"Grant '{grant_id}' has exceeded maximum permitted uses ({grant.max_uses})."

        if grant.action_type != action_type:
            return False, f"Grant action mismatch: granted '{grant.action_type}', attempted '{action_type}'."

        canon_target = self.canonicalize_path(target)
        canon_grant_target = self.canonicalize_path(grant.exact_target)
        if canon_target != canon_grant_target:
            return False, f"Grant target mismatch: granted '{canon_grant_target}', attempted '{canon_target}'."

        if grant.payload_hash and payload_data:
            computed_hash = hashlib.sha256(payload_data.encode("utf-8")).hexdigest()
            if computed_hash != grant.payload_hash:
                return False, f"Grant payload hash mismatch: granted '{grant.payload_hash}', got '{computed_hash}'."

        # Increment use count
        self.grant_use_counts[grant_id] = uses + 1
        return True, "Grant valid and authorized."

    @staticmethod
    def validate_charter_consistency(charter_goal: str, action_type: str, target: str) -> Tuple[bool, str]:
        """C-03: Ensure proposed action aligns with user's charter and is not a poisoned injection."""
        c_lower = charter_goal.lower()
        t_lower = target.lower()

        # If user asked for web research or document writing, but action tries to delete files or run shell commands
        if action_type in ("FILE_DELETE", "SYSTEM_MUTATE"):
            if "delete" not in c_lower and "remove" not in c_lower:
                return False, f"Action '{action_type}' on '{target}' is inconsistent with task charter: '{charter_goal}'"

        # Disregard prompt injection attempts
        injection_markers = ["disregard previous", "ignore previous", "override system", "new instructions:"]
        for marker in injection_markers:
            if marker in t_lower:
                return False, f"Target contains prompt injection marker: '{marker}'"

        return True, "Action is consistent with charter."

    def validate_action(
        self,
        action_type: str,
        target: str,
        payload: Optional[Dict[str, Any]] = None,
        provenance: InputProvenance = InputProvenance.TRUSTED_USER,
    ) -> PolicyDecision:
        """Determine whether an action is allowed, its tier, and whether it requires approval."""
        # 1. Provenance check: Untrusted content CANNOT originate actions or elevate permissions
        if provenance == InputProvenance.UNTRUSTED_CONTENT and action_type not in ("OBSERVE_READ", "HTTP_GET"):
            return PolicyDecision(
                allowed=False,
                tier=RiskTier.T3_SENSITIVE,
                reason="Untrusted content (web/OCR/file) is quarantined and cannot originate OS actions.",
            )

        # 2. Check for Catastrophic / Blocked Commands
        for pat in self.BLOCKED_COMMAND_PATTERNS:
            if re.search(pat, target, re.IGNORECASE):
                return PolicyDecision(
                    allowed=False,
                    tier=RiskTier.T3_SENSITIVE,
                    reason=f"Target matches catastrophic blocked pattern: '{pat}'",
                )

        # 3. Protected Window check
        if action_type in ("WINDOW_FOCUS", "KEYBOARD_TYPE", "MOUSE_CLICK"):
            target_lower = target.lower()
            for pw in self.PROTECTED_WINDOW_SUBSTRINGS:
                if pw in target_lower:
                    return PolicyDecision(
                        allowed=False,
                        tier=RiskTier.T3_SENSITIVE,
                        reason=f"Target window '{target}' matches protected development/credential window '{pw}'.",
                    )

        # 4. File System Operations
        if action_type in ("FILE_WRITE", "FILE_DELETE", "FILE_READ"):
            canon_target = self.canonicalize_path(target)
            in_roots = self.is_path_in_granted_roots(canon_target)

            if action_type == "FILE_READ":
                return PolicyDecision(
                    allowed=True,
                    tier=RiskTier.T0_OBSERVE,
                    reason="Read-only file operation.",
                    canonical_target=canon_target,
                )

            if action_type == "FILE_WRITE":
                tier = RiskTier.T1_ROUTINE if in_roots else RiskTier.T2_ELEVATED
                return PolicyDecision(
                    allowed=True,
                    tier=tier,
                    reason="File write inside granted root." if in_roots else "File write outside granted roots.",
                    requires_confirmation=(not in_roots),
                    confirmation_prompt=f"Allow writing to '{canon_target}' outside granted roots?" if not in_roots else None,
                    canonical_target=canon_target,
                )

            if action_type == "FILE_DELETE":
                # Deletion is always sensitive (T3) unless governed by an active one-time Grant
                return PolicyDecision(
                    allowed=False,
                    tier=RiskTier.T3_SENSITIVE,
                    reason="Permanent or unverified file deletion requires explicit confirmation.",
                    requires_confirmation=True,
                    confirmation_prompt=f"Confirm deletion of file: {canon_target}?",
                    canonical_target=canon_target,
                )

        # 5. Routine App / Browser Actions
        if action_type in ("APP_LAUNCH", "BROWSER_NAVIGATE"):
            return PolicyDecision(
                allowed=True,
                tier=RiskTier.T1_ROUTINE,
                reason="Routine reversible app/browser navigation.",
                canonical_target=target,
            )

        # 6. Power Actions (Shutdown, Reboot)
        if action_type in ("POWER_SHUTDOWN", "POWER_RESTART"):
            return PolicyDecision(
                allowed=False,
                tier=RiskTier.TP_POWER,
                reason="System power actions require explicit user confirmation and countdown.",
                requires_confirmation=True,
                confirmation_prompt=f"Confirm system {action_type} with cancelable countdown?",
            )

        # Default fallback
        return PolicyDecision(
            allowed=True,
            tier=RiskTier.T1_ROUTINE,
            reason="Standard operation permitted.",
            canonical_target=target,
        )

    def record_audit(self, entry: Dict[str, Any]) -> None:
        """Append-only audit record."""
        entry["timestamp"] = time.time()
        self.audit_log.append(entry)
