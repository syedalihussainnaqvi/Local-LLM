# ARCHITECTURAL ASSUMPTIONS & DESIGN DECISIONS

**Project**: Antigravity Local Bounded-Autonomy Computer-Operating Agent (v3)  
**Status**: Recorded for Phase 0 Capability Gate  

---

## 1. Hardware & Platform Constraints
1. **CPU Execution Environment**: The host system is an Intel Core i5-8265U (4 cores / 8 threads) with Intel UHD 620 integrated GPU and no NVIDIA CUDA GPU.
2. **RAM Budget**: Physical memory is 16 GB, with ~7.5 GB free. The base Qwen 2.5 1.5B model consumes ~3.3 GB RAM in memory. Total agent footprint must remain below 5.5 GB to prevent system paging and memory exhaustion.
3. **Storage Placement**: Drive `C:` has ~19.5 GB free space. Large caches, test logs, and disk-intensive artifacts should be monitored with strict rotation caps ($<500\text{ MB}$ total artifact retention).
4. **Display Architecture**: The primary display is 1920x1080 physical resolution running at 150% DPI scaling (yielding 1280x720 logical coordinate space in non-DPI-aware apps). All coordinate translation must explicitly map:
   $$\text{Physical Pixels} \longleftrightarrow \text{Logical Pixels} \longleftrightarrow \text{Normalized } [0, 1000]$$

---

## 2. Model & Latency Strategy
1. **Inference Latency Tradeoff**:
   - PyTorch FP32 on CPU achieves ~0.47 tokens/sec with TTFT $\sim 100$s when warming up from cold cache.
   - Therefore, the agent **CANNOT** invoke full LLM generation on every mouse click or routine routing check without freezing the user experience.
   - **Architecture**: A multi-tiered decision pipeline is assumed:
     - **Layer 1**: High-precision deterministic rules ($\ge 99\%$ precision) for standard operations (file listings, opening known apps, math, terminal tests). Latency: $<5\text{ ms}$.
     - **Layer 2**: Calibrated lightweight classifier / feature scoring for intent classification. Latency: $<50\text{ ms}$.
     - **Layer 3**: Volatility & temporal grounding check.
     - **Layer 4**: LLM generation with constrained decoding used for complex reasoning, planning, and cited synthesis.
2. **Voice Call Real-Time Budget**:
   - Voice conversation uses local streaming STT (Whisper INT8 / faster-whisper on CPU) and streaming TTS (Piper / Kokoro).
   - Voice scope for v1 is **local desktop voice call mode** (push-to-talk, call window overlay, system loopback echo cancellation). Remote telephony / SIP is explicitly out of scope.
   - Immediate canned voice acknowledgements ("On it", "Working on that") fire within $<400\text{ ms}$ while the planner calculates the execution graph.

---

## 3. Computer Control & Terminal Policy
1. **Protected Window Boundaries**:
   - The agent is strictly forbidden from interacting with:
     - The Antigravity IDE / VS Code / terminal running the agent itself.
     - Any password manager (1Password, Bitwarden, KeePass, Windows Credential Manager).
     - The agent's own overlay / call window (which must be marked `WS_EX_TRANSPARENT` / click-through and excluded from screen capture via `SetWindowDisplayAffinity`).
2. **Terminal Execution Contract**:
   - "Full terminal access" means a managed persistent PTY session (via `pywinpty`) operating with the user's standard privileges, strictly mediated by the Policy Gateway.
   - Commands are parsed into structured abstract syntax trees (executable + arguments); shell strings with `shell=True` are banned.
   - Dangerous commands (recursive deletion, format, user modifications) belong to Tier 3 and require explicit interactive confirmation or valid Grant tokens.

---

## 4. Testing & Safety Isolation
1. **Disposable Testing Ground**: All automated integration and E2E GUI tests will execute against dedicated fixture folders (`eval/fixtures/`), sandboxed mock browser pages, and stubbed power managers with canary files (`canary_safe.txt`).
2. **Reversible Mutations**: All file deletions default to the Windows Recycle Bin via `Send2Trash` rather than permanent unlinking.
