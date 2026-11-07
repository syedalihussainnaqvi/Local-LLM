# DISCOVERY REPORT (PHASE 0)
**Project**: Antigravity Local Bounded-Autonomy Computer-Operating Agent with Voice & Terminal  
**Timestamp**: 2026-10-22  
**Baseline Branch**: `feat/antigravity-master-v3` (created from initial commit `a73e17b`)  
**Status**: [VERIFIED] Empirical Audit Complete  

---

## 1. Environment Audit (Section 1.A)

| Component | Detected Specification | Verification Status |
| :--- | :--- | :--- |
| **OS & Version** | Microsoft Windows 11 Home / Pro (`10.0.26300.0`) | [VERIFIED] PowerShell CIM |
| **User Elevation** | Normal User (`IsInRole(Administrator) == False`) | [VERIFIED] PowerShell .NET |
| **Locale & Timezone** | `en-US` (System Locale: `en-US`, Current UTC+05:00) | [VERIFIED] Culture check |
| **CPU** | Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz (4 Cores, 8 Logical Processors) | [VERIFIED] Win32_Processor |
| **RAM** | 16,661,812 KB (~16 GB) Total Visible, 7,546,892 KB (~7.5 GB) Free Physical | [VERIFIED] Win32_OperatingSystem |
| **GPU / VRAM** | Intel(R) UHD Graphics 620 (1 GB Shared RAM). **No dedicated NVIDIA GPU / No CUDA** (`nvidia-smi` not found). | [VERIFIED] Win32_VideoController |
| **Storage / Disk Free**| `C:` 19.55 GB free (139 GB used) — *Capacity warning: C is constrained*<br>`D:` 141.07 GB free<br>`E:` 117.93 GB free | [VERIFIED] PSDrive FileSystem |
| **Display & DPI** | Physical: 1920x1080 @ 150% scaling (`DPI_AWARENESS_PER_MONITOR_AWARE`). Logical without DPI awareness: 1280x720. 32-bit color. Single monitor (`\\.\DISPLAY1`). | [VERIFIED] Win32 GDI & Forms |
| **Runtimes** | Python 3.14.7, pip 26.2.1, Node.js v24.19.0, Git 2.55.0.windows.4 | [VERIFIED] CLI inspection |
| **Virtualenv Status** | System Python environment (`C:\Users\Jahan\AppData\Local\Python\pythoncore-3.14-64`). No active venv. | [VERIFIED] pip list |
| **Audio Hardware** | Realtek High Definition Audio, Intel(R) Display Audio | [VERIFIED] Win32_SoundDevice |

---

## 2. Model & Inference Engine Audit (Section 1.B)

- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct` (~3.1 GB base weights cached in `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct`).
- **Adapter**: LoRA adapter in `models/adapter/` (`adapter_config.json`, `adapter_model.safetensors` size: 73.9 MB).
  - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
  - Rank $r=16$, $\alpha=32$, dropout $0.1$.
- **Training Provenance (`models/adapter/run_info.json`)**:
  - Dataset: `OpenAssistant/oasst1` (3,097 train examples, 296 val examples, 297 test examples).
  - **CRITICAL DEFECT FINDING**: The adapter was fine-tuned purely on conversational assistant text (`oasst1`), **not on tool calling, function calling, or GUI grounding**. It contains zero OS automation training tokens.
- **Inference Runtime (`src/local_llm/llm_engine.py`)**:
  - Uses `transformers.AutoModelForCausalLM` + `peft.PeftModel` on CPU (`torch.device("cpu")`).
  - Threading: `torch.set_num_threads(8)`.
- **Empirical CPU Benchmark Results**:
  - Model load time: **67.74 seconds**.
  - Process RAM after model load: **3,309.2 MB**.
  - Time-To-First-Token (TTFT): **106.67 seconds**.
  - Generation speed: **0.47 tokens/second** on Intel Core i5-8265U CPU!
  - Tool JSON Emission: Valid JSON was emitted (`{"action": "open_app", "app": "Google Chrome"}`), but took **30.66 seconds**.
  - **Conclusion**: Pure unquantized PyTorch FP32/BF16 on an Intel i5-8265U CPU cannot deliver real-time voice latency (p50 $\le$ 1.0s) or rapid step loops without INT8/INT4 quantization (GGUF / llama.cpp or ONNX Runtime) and high-precision routing that short-circuits LLM generation for deterministic tasks.

---

## 3. Tool Layer Audit (Section 1.C)

- **Registry**: Dispersed across modules (`src/local_llm/desktop_agent.py`, `src/local_llm/web_scraper.py`, `src/local_llm/social_scraper.py`, `src/local_llm/script_engine.py`, `src/local_llm/trending_engine.py`).
- **Schema & Tiers**: Informal dataclasses in `src/local_llm/desktop_agent.py` (`RiskLevel.SAFE`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), lacking typed Pydantic validation, structured Grant tokens, and out-of-process gatekeeper enforcement.
- **Tool-Calling Protocol**: `models/adapter/chat_template.jinja` includes Qwen `<tool_call>` syntax, but `src/local_llm/llm_engine.py` does not constrain decoding, leading to potential parser failures and non-deterministic text generation.

---

## 4. Web & Social Scraper Audit (Sections 1.D & 1.E)

### Web Scraper (`src/local_llm/web_scraper.py`)
- **Libraries**: `requests` + `BeautifulSoup4`.
- **Search Provider**: `authentic_composite` (composite of DuckDuckGo HTML, Google News RSS, Wikipedia API, Brave fallback).
- **SSRF Defenses**: DNS lookup check `validate_url_safety()` exists, but:
  - Fails to block private IPv6 scopes completely.
  - Redirect loop tracking does not re-resolve DNS on dynamically rebound DNS hostnames.
  - Port blocking is a small allowlist rather than default-deny on non-standard ports.
- **JS Rendering**: No headless browser or rendering pipeline; JS-heavy SPAs return empty shells.

### Social Scraper (`src/local_llm/social_scraper.py`)
- **Coverage**: TikTok, X/Twitter, Instagram, YouTube, Reddit, LinkedIn, GitHub.
- **Risk**: Unofficial DOM/HTML scraping without authentication. Frequent breaking changes, anti-bot Cloudflare blocks, and Terms-of-Service violation risks on Instagram and X. Must separate read-only RSS/public queries from automated actions and provide clear compliance fallbacks.

---

## 5. Computer Control Inventory (Section 1.F)

- **Capture**: `PIL.ImageGrab.grab()` in `src/local_llm/desktop_agent.py` failed with `OSError: screen grab failed` due to missing per-monitor DPI awareness in the Python process.
  - *Empirically Fixed*: Native Win32 GDI BitBlt with `SetProcessDpiAwareness(2)` successfully captures the full physical `1920x1080` screen in $<25\text{ ms}$.
- **Mouse / Keyboard**: `pyautogui` clicks and keystrokes run in-process without verifying foreground window focus before typing (Defects C-07, C-08).
- **Kill Switch**: `pyautogui.FAILSAFE` exists, but is strictly in-process and inert if the agent thread hangs (Defect C-02).
- **Execution**: Direct `subprocess.Popen(..., shell=True)` present in `open_app()` and `terminal` execution (Defect C-05).

---

## 6. Baseline Test Suite Results

Historical baseline before the `src/` package layout: ran the existing suite via `python -m unittest discover -s tests`:
- Total tests: **40**
- Passed: **39**
- Failed: **1**
  - `FAIL: test_question_automatic_web_scraping (test_query_router.TestQueryRouter)`
  - Query: `"Explain how quantum computers use qubits for computation."`
  - Expected: `live_web` | Actual: `llm_only`
  - Root Cause: Uncalibrated keyword matching in `src/local_llm/query_router.py`.

---

## 7. Ranked Root Causes (Section 1.G)

1. **Uncalibrated Heuristic Routing (H-30, H-29)**: `src/local_llm/query_router.py` relies on brittle keyword lists (`is_trending_query`, `CODING_OR_GENERATIVE_KEYWORDS`). Queries with overlapping keywords either over-call web scrapers (causing slow responses) or fail to retrieve fresh information when required.
2. **Missing Out-of-Process Safety Architecture (C-01, C-02, C-05)**: Tools in `src/local_llm/desktop_agent.py` can run arbitrary shell commands (`shell=True`), mutate files, or launch processes without an out-of-process supervisor, without checking the foreground window, and without cryptographic/structured Grant objects.
3. **Hardware-Inference Bottleneck (H-36, H-55)**: On the Intel Core i5-8265U CPU (no dedicated GPU), FP32 PyTorch inference runs at 0.47 tokens/sec with TTFT > 100 seconds. Relying on continuous LLM generation for routine tool selection and voice conversation will breach all latency budgets unless Layer 1/2 routing is deterministic and INT8 quantized runtimes are deployed.
