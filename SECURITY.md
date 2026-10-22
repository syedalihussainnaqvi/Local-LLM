# SECURITY & SAFETY ARCHITECTURE

**Project**: Antigravity Local Bounded-Autonomy Computer-Operating Agent (v3)  
**Status**: [VERIFIED] Active Security Gates in Place  

---

## 1. Provenance Tagging & Injection Defense (C-03, H-60)

Text inside the system is categorized by origin into two distinct provenance classes:
1. `TRUSTED_USER`: Commands originating from the user's direct interactive input (keyboard prompt, explicit voice instruction confirmed by user, task charter).
2. `UNTRUSTED_CONTENT`: Web page text, scraped documents, OCR output from screenshots, chat replies, and shell/command outputs.

**Invariant**: Untrusted text is strictly quarantined. It can be summarized or analyzed by a constrained prompt, but it can **NEVER** originate OS actions, raise permission tiers, modify the task charter, or create Grants.

---

## 2. Path Canonicalization & Root Boundaries (C-05, M-27)

To defeat path traversal, symlink swaps, 8.3 short name tricks, and UNC evasion:
- All paths are fully resolved via `pathlib.Path.resolve()` and `os.path.realpath()`.
- Windows environment variables and user home shortcuts are expanded prior to policy evaluation.
- Known folders (`Documents`, `Desktop`, `Downloads`) are resolved via OS Known-Folder APIs (accounting for OneDrive redirection).
- Target paths outside granted roots are denied by default or require elevated Tier 2 confirmation.

---

## 3. Catastrophic Command Prevention (C-05)

The following command patterns are unconditionally blocked by the Policy Gateway:
- Unbounded disk formats: `format <drive>:`
- Unbounded recursive deletions targeting system roots: `del /s /q C:\...`, `rmdir /s /q C:\...`, `Remove-Item -Recurse C:\...`
- Obfuscated/encoded scripts: `powershell -EncodedCommand ...`
- Boot and partition tampering: `bcdedit`, `diskpart`, `reg delete HKLM`

---

## 4. SSRF & Network Protections (C-04)

- Scheme allowlist strictly limited to `http` and `https`.
- Pre-fetch DNS resolution: hostname is resolved and verified against private, loopback, multicast, link-local, and cloud metadata (`169.254.169.254`, `100.100.100.200`) IP ranges.
- Dynamic DNS re-checking across redirect hops.
- Response size capped at 5 MB by default with streaming byte counters to defeat decompression bombs.

---

## 5. Voice & Terminal Safeguards (C-50, C-51, C-52)

- Voice transcripts are treated as untrusted for authorization: spoken words can never grant permissions or raise tiers.
- High-impact actions (Tier 3 or destructive operations) require confirmation via a non-voice channel (hotkey or on-screen button).
- Terminal sessions run persistent ConPTY/WinPTY sessions under the user's standard privileges with command AST parsing. Direct `shell=True` execution is prohibited.
