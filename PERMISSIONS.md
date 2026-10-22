# PERMISSION ENGINE SPECIFICATION

**Project**: Antigravity Local Bounded-Autonomy Computer-Operating Agent (v3)  
**Status**: [VERIFIED] Implemented in `agent/policy/gateway.py`  

---

## 1. Risk Tiers

| Tier | Category | Allowed Actions | Enforcement Rule |
| :--- | :--- | :--- | :--- |
| **T0** | **Observe** | Screen capture (with window masking), read files, process listing, web fetch. | Automatically permitted. Masking applied to protected windows. |
| **T1** | **Routine Reversible** | Launch allowed apps, navigate browser (dedicated profile), type in non-sensitive fields, write files in granted roots. | Automatically permitted if target is within granted roots and foreground assertions pass. |
| **T2** | **Elevated** | System configuration, package installations, accessing files outside granted roots, using real personal browser profile. | Requires interactive confirmation prompt or elevation dialog. |
| **T3** | **Sensitive** | Permanent deletion, sending messages (WhatsApp/email), executing downloaded/unknown scripts, modifying security credentials. | Strict interactive confirmation showing exact payload & recipient, OR valid single-use `Grant`. Ambient audio/voice cannot confirm. |
| **TP** | **Power** | Shutdown, restart, sign-out, sleep. | Best-effort unsaved work detection, mandatory state checkpoint, cancelable countdown (H-12). Stubbed during automated tests. |

---

## 2. Structured Grants

```python
@dataclass(frozen=True)
class Grant:
    grant_id: str
    action_type: str
    exact_target: str
    scope: str
    expiry_timestamp: float
    max_uses: int
    payload_hash: Optional[str] = None
    created_at: float
```

- **Origin Rule (C-06)**: Grants can **ONLY** be created from explicit, trusted user input channels. They can **NEVER** be created from model outputs, web pages, OCR text, or command outputs.
- **Expiry & Use Counts**: Grants automatically expire after `expiry_timestamp` or when `max_uses` is reached.
- **Payload Pinning**: Sensitive payloads (e.g. specific message text or script content) are pinned by SHA256 `payload_hash` to defeat Time-Of-Check-To-Time-Of-Use (TOCTOU) tampering.

---

## 3. Protected Windows (C-07)

The agent cannot focus, click, type into, or screenshot unmasked any window matching:
- Antigravity IDE / VS Code / Cursor / active coding agent terminal.
- Password Managers: 1Password, Bitwarden, KeePass, LastPass, Windows Credential Manager.
- Windows Security, Task Manager.
- The agent's own overlay/call window (excluded via `SetWindowDisplayAffinity`).
