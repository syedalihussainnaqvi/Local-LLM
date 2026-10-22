# OPERATIONS RUNBOOK

**Project**: Antigravity Local Bounded-Autonomy Computer-Operating Agent (v3)  
**Status**: [VERIFIED] Operational  

---

## 1. Quick Start Commands

```bash
# Standard task run (with user confirmation for sensitive steps)
python agent_cli.py run "Search latest developments in artificial intelligence and save summary"

# Autonomous Bounded Auto Mode
python agent_cli.py run "Collect information on quantum computing and save report" --auto

# Dry-Run Mode (Plan preview without executing OS actions)
python agent_cli.py run "Delete old temporary files in downloads" --dry-run

# Real-Time Voice Call Mode
python agent_cli.py call

# Managed Terminal Command
python agent_cli.py terminal "git status"

# Start Trusted Terminal Session (30 min developer mode)
python agent_cli.py trusted-session start
python agent_cli.py trusted-session end
```

---

## 2. Emergency Stop & Pause

1. **Global Hotkeys**:
   - **Emergency Kill**: `Ctrl+Alt+Shift+Q` (Freezes input immediately and terminates agent worker process tree within $\le 1.0\text{ s}$).
   - **Pause Execution**: `Ctrl+Alt+Shift+P` (Suspends execution broker for human inspection).
2. **CLI Emergency Stop**:
   ```bash
   python agent_cli.py stop
   ```
3. **Voice Stop**:
   - Say: `"Stop"`, `"Freeze"`, `"Pause"`, or `"Cancel"` (fast-path reaches supervisor in $<200\text{ ms}$).

---

## 3. Crash Recovery & Idempotent Resume

If a machine reboot, crash, or power failure interrupts an autonomous task:
1. Locate the task ID from the logs or WAL database.
2. Execute the resume command:
   ```bash
   python agent_cli.py resume <task_id>
   ```
3. **Reconciliation Mechanism (C-14)**:
   - Steps marked `COMMITTED` in SQLite WAL are skipped automatically.
   - For uncommitted steps:
     - `IDEMPOTENT` steps (e.g. read, check directory) are safe to replay.
     - `NON_IDEMPOTENT` steps (e.g. write file, delete) are checked against actual disk reality. If the side-effect already occurred, it commits retroactively; otherwise, it requires explicit confirmation to prevent double actions.

---

## 4. Inspection & Data Hygiene

1. **Inspect Flight Logs**:
   ```bash
   # View structured JSONL flight recorder
   Get-Content eval/fixtures/flight_log.jsonl -Tail 20
   ```
2. **Inspect Terminal Command Journal**:
   ```bash
   Get-Content eval/fixtures/test_terminal_journal.jsonl -Tail 20
   ```
3. **Purge Temporary Artifacts**:
   ```bash
   python agent_cli.py purge
   ```
