# TERMINAL ACCESS & COMMAND POLICY

**Project**: Antigravity Local Bounded-Autonomy Computer-Operating Agent (v3)  
**Status**: [VERIFIED] Implemented in `agent/tools/terminal.py`  

---

## 1. Terminal Architecture (Section 6B)

The agent operates persistent terminal sessions running under standard user privileges. All command proposals are parsed, classified, and mediated by the Policy Gateway before execution.

```
Model Proposal -> Command AST Parser -> Policy Gateway -> T1 Auto / T2 Prompt / T3 Strict -> Persistent Shell -> Unsanitized Output Tagged Untrusted -> JSONL Journal
```

---

## 2. Command Tiers

| Tier | Category | Example Commands | Approval Requirement |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **Read-Only / Project Inspection** | `git status`, `dir`, `ls`, `type`, `cat`, `python --version`, `pytest`, `npm test` | Automatically allowed within granted roots. |
| **Tier 2** | **Installs / Global Changes** | `pip install <pkg>`, `npm install <pkg>`, network downloads, modifying files outside granted roots | Interactive confirmation prompt, OR active Trusted Terminal Session. |
| **Tier 3** | **Sensitive / Destructive** | `del <file>`, `rm <file>`, `Remove-Item`, `net user`, `reg delete`, `shutdown` | Explicit confirmation showing exact target and command. Trusted Session CANNOT override. |
| **BLOCKED**| **Catastrophic** | `del /s /q C:\`, `rmdir /s /q C:\`, `format <drive>:`, `bcdedit`, `diskpart`, `powershell -EncodedCommand` | Unconditionally blocked. |

---

## 3. Trusted Terminal Session (Opt-In Developer Mode)

A time-limited, directory-scoped token that allows developer workflows (e.g. running builds, test suites, and project-local package installations) without repeated prompts:
- **Enable via CLI**: `python agent_cli.py trusted-session start`
- **Revoke via CLI / Voice**: `python agent_cli.py trusted-session end` (or speak "end trusted session")
- **Hard Expiry**: 30 minutes default; never survives agent restart.
- **Safety Boundary**: Tier 3 commands and catastrophic blocks are **never** bypassed by a trusted session.

---

## 4. Environment Hygiene & Untrusted Output (H-59, H-60)
- **Secret Scrubbing**: API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AWS_SECRET_ACCESS_KEY`) are scrubbed from subshell environments.
- **Output Provenance**: All terminal stdout and stderr is tagged with `InputProvenance.UNTRUSTED_CONTENT`. It is strictly forbidden from originating actions or elevating security grants.
- **Monotonic Timeout**: Every command executes under a monotonic deadline; hung commands have their entire process tree forcefully terminated.
