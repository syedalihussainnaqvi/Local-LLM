# RISK REGISTER & KNOWN LIMITATIONS

**Project**: Antigravity Local Bounded-Autonomy Computer-Operating Agent (v3)  
**Status**: Continuously Maintained (Section 16)  

---

## 1. Risk Register

| Risk ID | Risk Description | Likelihood | Impact | Mitigation Strategy | Owner |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **R-01** | **Wrong-Window Input** (typing into IDE, terminal, or password manager) | Low | High | `assert_foreground_window()` checks title pattern immediately before every keystroke batch; protected window blocklist aborts input. | Input Broker |
| **R-02** | **Accidental Data Loss** (permanent file deletion) | Low | Critical | Deletions route to Windows Recycle Bin via `Send2Trash`; permanent deletes classified as Tier 3 requiring interactive confirmation. | Policy Gateway |
| **R-03** | **Prompt Injection via Web/OCR** | Med | Critical | Provenance tagging separates trusted user charter from untrusted web/OCR/file content; `validate_charter_consistency` rejects unauthorized commands. | Policy Gateway |
| **R-04** | **SSRF & Cloud Metadata Access** | Low | High | Pre-fetch DNS resolution blocks private, loopback, link-local, and cloud metadata IPs (`169.254.169.254`, `100.100.100.200`); manual redirect re-checking. | Web Scraper |
| **R-05** | **Model Hallucination in Verification** | Med | High | Programmatic verification first (file exists, size, SHA256, HTTP status, process check). "Click sent" is never counted as verification. | Verifier |
| **R-06** | **GUI Flakiness Across App Updates** | Med | Med | Accessibility-first UI Automation hierarchy, fallback caching, and Failure Ledger to prevent repeating failed locators. | Desktop Control |
| **R-07** | **RAM / Memory Exhaustion** (CPU-only host) | Low | High | Residency budget caps resident models to $\sim 3.5\text{ GB}$ within 7.5 GB free physical RAM; process memory limits monitored. | Supervisor |
| **R-08** | **Platform Terms Violations (Social)** | Med | Med | Official public endpoints preferred; rate-limiting; compliance warnings; no paywall or bot bypass; unverified claim labeling. | Social Adapters |
| **R-09** | **Credential Exposure in Screenshots** | Low | High | Dynamic window masking blacks out bounding boxes of password managers and protected IDE windows prior to vision processing. | Screen Capturer |
| **R-10** | **Resume Duplication on Crash** | Low | Critical | SQLite Write-Ahead Log writes `INTENT` before action and `COMMITTED` after; crash reconciliation skips committed steps and queries user on uncommitted non-idempotent steps. | WAL Manager |
| **R-11** | **Resource Leaks in Long Runs** | Low | Med | Native GDI device contexts and SQLite connections closed via context managers; temporary files purged automatically. | Auto Controller |
| **R-12** | **Model License Constraints** | Low | Low | Qwen 2.5 1.5B Instruct model released under Apache 2.0 permissive license allowing local deployment and modifications. | AI Engineer |
| **R-13** | **Voice Self-Trigger & TV Audio Loops** | Low | High | System-audio loopback echo cancellation; mic gating during playback; push-to-talk default; spoken text cannot create grants. | Voice Pipeline |
| **R-14** | **STT Misrecognition of Destructive Commands** | Med | High | Voice never executes raw transcripts as shell strings; Tier 2+ and Tier 3 actions require explicit non-voice confirmation. | Voice Pipeline |
| **R-15** | **Microphone Privacy Risk** | Low | High | Local-only execution; active mute toggle provably drops audio frames; zero persisted raw audio bytes on disk. | Voice Pipeline |
| **R-16** | **Latency Budget Miss on CPU Hardware** | High | Med | Deterministic Layer 1 rules and Layer 2 calibrated classifier short-circuit 90%+ of queries, bypassing slow CPU LLM inference. | Router |
| **R-17** | **Terminal Obfuscation & Output Injection** | Low | Critical | Command AST parser blocks encoded commands (`-EncodedCommand`); terminal output tagged as untrusted provenance. | Terminal |
| **R-18** | **Trusted Terminal Session Scope Creep** | Low | High | Strict time limit (30m), directory tree scoping, automatic expiry, and inability to override Tier 3 or catastrophic blocks. | Terminal |

---

## 2. Documented Hard Limits (Honest Limitations)
1. **CAPTCHAs & Biometric Prompts**: The agent cannot solve CAPTCHAs, Windows Hello facial/fingerprint prompts, or UAC secure desktops; it detects them and pauses execution, handing control to the user.
2. **Elevated Windows**: Non-elevated agent cannot inject input into elevated administrative windows due to Windows User Interface Privilege Isolation (UIPI). It prompts the user to handle elevation.
3. **CPU Generation Speed**: Unquantized PyTorch float32 generation on the Intel Core i5-8265U CPU achieves $\sim 0.5\text{ tokens/sec}$. Heavy multi-turn voice reasoning requires INT8 quantization (GGUF / llama.cpp / ONNX) to hit real-time response budgets.
