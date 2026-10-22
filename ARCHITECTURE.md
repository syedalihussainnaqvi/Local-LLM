# SYSTEM ARCHITECTURE

**Project**: Antigravity Local Bounded-Autonomy Computer-Operating Agent (v3)  
**Status**: [VERIFIED] Phase 1 Walking Skeleton Active  

---

## 1. Process & Concurrency Model

```mermaid
graph TD
    User([User Voice / Text]) --> UI[Overlay / Call Window]
    User --> SupHotkey[Emergency Hotkey: Ctrl+Alt+Shift+Q]

    subgraph SupervisorProcess [Supervisor Process - Out-of-Process Watchdog]
        SupHotkey --> Watchdog[Supervisor / Watchdog]
        Watchdog -. Heartbeat Monitor .-> AgentProc
        Watchdog -. Kill / Freeze Signal .-> InputBroker
        Watchdog -. Immediate Terminate .-> AgentProc
    end

    subgraph AgentProcess [Agent Actor Process]
        UI --> Dialogue[Dialogue / Input Parser]
        Dialogue --> Core[Core Engine]
        Core --> Router[Multi-Tier Router]
        Core --> Planner[Deep Task Planner]
        Core --> WAL[SQLite WAL State Engine]

        Core --> PolicyGateway{Policy Gateway}
        
        PolicyGateway -->|Allowed T0/T1| Tools[Tool Registry]
        PolicyGateway -->|T2/T3/TP| Approval[Interactive User Approval / Grants]
        
        Tools --> InputBroker[Input Broker]
        Tools --> Browser[Browser Tool - Dedicated Profile]
        Tools --> FileOps[Atomic File Ops]
        Tools --> Terminal[Persistent WinPTY Terminal]
    end

    subgraph HardwareOS [Operating System & Hardware]
        InputBroker --> Display[Physical 1920x1080 @ 150% DPI]
        InputBroker --> KeyboardMouse[Windows SendInput Unicode / GDI]
        Terminal --> ConPTY[Windows ConPTY Session]
    end
```

### Key Components

1. **Supervisor Process (`agent/supervisor/watchdog.py`)**:
   - Runs out-of-process from the agent actor.
   - Listens to global hotkeys (`Ctrl+Alt+Shift+Q` for emergency stop, `Ctrl+Alt+Shift+P` for pause).
   - Monitors heartbeats; can freeze input or forcefully terminate the agent process tree within $<1.0\text{ s}$ stop latency.
2. **Policy Gateway (`agent/policy/gateway.py`)**:
   - The **ONLY** entry point through which tools can interact with the operating system.
   - Enforces risk tiers (T0 Observe, T1 Routine, T2 Elevated, T3 Sensitive, TP Power).
   - Validates provenance: untrusted content (web/OCR/file text) can never originate actions or create Grants.
   - Canonicalizes all paths (symlinks, junctions, 8.3 short names, UNC) before policy evaluation.
   - Blocks catastrophic command patterns and protected windows.
3. **Write-Ahead Log (`agent/core/wal.py`)**:
   - SQLite WAL state machine.
   - Logs `INTENT` record prior to action execution and `COMMITTED` record after programmatic verification.
   - Distinguishes `IDEMPOTENT` from `NON_IDEMPOTENT` steps for crash recovery.
4. **Input Broker (`agent/core/input_broker.py`)**:
   - Single serialized owner of mouse and keyboard.
   - Asserts active foreground window before every keystroke batch (`assert_foreground_window()`).
   - Rejects input directed at protected development windows or password managers.
   - Maps physical coordinates (1920x1080) to logical coordinates (1280x720) across DPI scaling.
5. **Tool Registry & Subsystems**:
   - `browser.py`: Google Chrome launched with a dedicated, isolated agent profile (`%LOCALAPPDATA%\AntigravityAgent\BrowserProfile`).
   - `file_ops.py`: Atomic write operations with cryptographic SHA256 checksums and size verification.
