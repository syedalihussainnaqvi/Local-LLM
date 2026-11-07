# PHASE 0 CAPABILITY GATE & ARCHITECTURAL DECISION REPORT

**Project**: Antigravity Local Bounded-Autonomy Computer-Operating Agent (v3)  
**Date**: 2026-10-22  
**Decision**: **GO** (Proceed to Phase 1 Walking Skeleton under Stated Hardware Mitigations)  
**Blockers**: **NONE** requiring immediate stop; key hardware trade-offs documented below.

---

## 1. Key Empirical Findings Summary

1. **Hardware Profile**:
   - Intel Core i5-8265U (4C / 8T), 16 GB RAM (7.5 GB free), Intel UHD 620 integrated GPU, **No CUDA GPU**.
   - Model memory residency: Qwen2.5-1.5B base + LoRA adapter occupies **3.31 GB RAM**.
   - Generation throughput on CPU: **0.47 tokens/second** (TTFT: 106.67 s cold-start, 30.6 s for short tool JSON).
2. **Fine-Tuning Dataset Reality**:
   - `models/adapter/run_info.json` proves training was performed on `OpenAssistant/oasst1` (dialogue), with **zero tool-calling or GUI-grounding examples**.
   - *Implication*: The model cannot be trusted to self-route or ground coordinates reliably without a strict Policy Gateway, deterministic Layer 1/2 routing, and structured JSON-schema constrained decoding.
3. **Screen Capture & DPI Resolution**:
   - Standard `ImageGrab.grab()` fails on this Windows environment due to non-DPI-aware process contexts.
   - Native Win32 GDI capture with `SetProcessDpiAwareness(2)` succeeds at full 1920x1080 resolution in $<25\text{ ms}$.

---

## 2. Model Residency & VRAM/RAM Budget

Since there is no dedicated VRAM, all models reside in system RAM (Total available: ~7.5 GB free):

| Component | Engine | Format / Precision | Allocated RAM | Active State |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5 1.5B + LoRA** | PyTorch / GGUF CPU | INT8 / FP16 | ~1.8 – 3.3 GB | Resident in Agent Core |
| **Streaming STT** | Faster-Whisper / Whisper.cpp | INT8 (tiny.en / base.en) | ~250 – 400 MB | Resident during Voice Call |
| **Streaming TTS** | Piper / Kokoro | ONNX / INT8 | ~180 – 300 MB | Resident during Voice Call |
| **VAD Engine** | Silero VAD | ONNX | ~30 MB | Continuous background |
| **Total Resident Memory** | | | **~2.8 – 4.0 GB** | **Well within 7.5 GB free limit** |

---

## 3. Per-Step Latency Budget (Target vs Reality)

| Stage | Target Budget | Pure CPU LLM (Unmitigated) | Mitigated Architecture (Phase 1-9) |
| :--- | :--- | :--- | :--- |
| **Screen Capture** | $\le 50\text{ ms}$ | $25\text{ ms}$ (Win32 GDI) | **$25\text{ ms}$** [VERIFIED] |
| **Perception / OCR** | $\le 200\text{ ms}$ | $300\text{ ms}$ | **UI Automation / Windows Accessibility API first** |
| **Tool Selection** | $\le 100\text{ ms}$ | $30,000\text{ ms}$ (LLM gen) | **$<10\text{ ms}$ (Layer 1 Rules + Layer 2 Classifier)** |
| **Execution** | $\le 300\text{ ms}$ | $200\text{ ms}$ | **$150\text{ ms}$ (Input Broker + WinPTY)** |
| **Verification** | $\le 150\text{ ms}$ | $10,000\text{ ms}$ | **$<20\text{ ms}$ (Programmatic: file/process/UIA)** |
| **Total Step Latency** | **$\le 800\text{ ms}$** | $>40,000\text{ ms}$ | **$\le 400 - 800\text{ ms}$ for routine steps** |

---

## 4. Phase 1 Implementation Plan (Walking Skeleton)

With Phase 0 complete and verified, we proceed to **Phase 1: WALKING SKELETON**:
1. **Out-of-Process Supervisor (`src/local_llm/agent/supervisor/watchdog.py`)**:
   - Spawns in an independent process.
   - Listens to emergency-stop hotkey (`Ctrl+Alt+Shift+Q`).
   - Owns the heartbeat watchdog and can terminate/freeze input immediately ($<1\text{ s}$ stop latency).
2. **Minimal Policy Gateway (`src/local_llm/agent/policy/gateway.py`)**:
   - Single choke-point for all OS actions.
   - Implements T0, T1, T2, T3 tier classification.
   - Path canonicalization and default-deny outside granted roots.
3. **Write-Ahead Log (WAL) State Engine (`src/local_llm/agent/core/wal.py`)**:
   - SQLite WAL state machine recording `INTENT` before action and `COMMITTED` after action.
4. **Input Broker (`src/local_llm/agent/core/input_broker.py`)**:
   - Single owner of synthetic input with coordinate scaling (1920x1080) and foreground window assertion.
5. **Thin Vertical Slice Verification**:
   - Run end-to-end task: Propose search $\rightarrow$ Policy Gateway approvals $\rightarrow$ Open Chrome (agent profile) $\rightarrow$ Save text report to fixture folder $\rightarrow$ Programmatic verification.
   - Run in the disposable test harness with canary validation.
