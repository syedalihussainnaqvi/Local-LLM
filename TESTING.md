# TESTING & VERIFICATION METHODOLOGY

**Project**: Antigravity Local Bounded-Autonomy Computer-Operating Agent (v3)  
**Status**: [VERIFIED] 89 Tests Passing Cleanly (88 passing + 1 fixed)

---

## 1. Test Pyramid & Execution Structure

```
                  /\
                 /E2E\       -> eval/harness/test_walking_skeleton.py (Thin E2E Slice)
                /------\
               /Security\    -> tests/security/test_permission_adversarial.py
              /----------\
             /Integration \  -> tests/unit/test_execution_engine.py, test_auto_mode.py
            /--------------\
           /   Unit Tests   \ -> tests/unit/ (router, web, social, voice, terminal, llm)
          /------------------\
```

### Full Test Inventory (89 total)

| Module | File | Tests |
|---|---|---|
| Auto Mode (bounded autonomy) | `tests/unit/test_auto_mode.py` | 4 |
| Desktop Control (screen, DPI, power) | `tests/unit/test_desktop_control.py` | 4 |
| Execution Engine (WAL, crash recovery) | `tests/unit/test_execution_engine.py` | 4 |
| Hardened Web (SSRF, SimHash) | `tests/unit/test_hardened_web.py` | 5 |
| LLM Client Hardening | `tests/unit/test_llm_client_hardening.py` | 6 |
| Longform Writer | `tests/unit/test_longform_writer.py` | 8 |
| Rich Response Prompts | `tests/unit/test_rich_response.py` | 8 |
| Router Calibration | `tests/unit/test_router_calibration.py` | 6 |
| Social Adapters | `tests/unit/test_social_adapters.py` | 4 |
| Terminal Subsystem | `tests/unit/test_terminal_subsystem.py` | 5 |
| Voice Pipeline (H-53..H-58) | `tests/unit/test_voice_pipeline.py` | 6 |
| Voice Bugfix Stress + Doctor | `tests/unit/test_voice_bugfix.py` | 8 |
| Voice Web API Endpoints | `tests/unit/test_voice_web_api.py` | 6 |
| Walking Skeleton E2E | `eval/harness/test_walking_skeleton.py` | 8 |
| Permission Adversarial Security | `tests/security/test_permission_adversarial.py` | 7 |

---

## 2. Statistical Rigor (H-40, H-41, M-45)

- **Wilson Score 95% Confidence Intervals**: Evaluated on routing and tool-decision datasets:
  $$\hat{p} = \frac{k}{n}, \quad \text{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
- **Unnecessary Tool Call Rate**: 0% on benchmark prompts (Wilson upper bound $\le 5\%$).
- **Missed Tool Rate**: 0% on fresh 2025/2026 temporal prompts.

---

## 3. Disposable Test Environment (C-39)

- All tests execute strictly against `eval/fixtures/`.
- **Canary File Invariant**: `eval/fixtures/canary.txt` is verified intact before and after test suites. Any accidental out-of-bounds mutation fails immediately.
- Real user directories (`Documents`, `Desktop`, `Downloads`) are protected and never mutated during automated test execution.

---

## 4. Known Fixes Applied

### TTS SAPI COM Flakiness Fix (test_01_injected_wav_20_runs)

**Root Cause**: Windows `System.Speech.Synthesis.SpeechSynthesizer` is a COM singleton that
fails silently under rapid-successive 20-run stress tests. The prior code had only 2 retries
with 6s/5s timeouts and no backoff, resulting in intermittent 0-byte output files.

**Fix** ([`agent/voice/audio_engine.py`](agent/voice/audio_engine.py)):
- 3 retry attempts (was 2)
- 250ms / 500ms staggered backoff between retries (SAPI COM cooldown)
- 150ms filesystem-flush sleep after subprocess exits (SAPI writes asynchronously on some Windows builds)
- Stderr logging on silent failure for easier diagnosis

**Fix** ([`agent/voice/pipeline.py`](agent/voice/pipeline.py)):
- Replaced `int(time.time()*1000)` timestamp filename with `os.getpid()_time.monotonic_ns()` to guarantee filename uniqueness under rapid concurrent calls
- Added cleanup of temp WAV after TTS bytes are recorded to prevent `eval/fixtures/` disk accumulation

---

## 5. Test Suite Execution Command

To run the complete verified test suite across all subsystems:

```bash
python -m pytest tests/unit/ eval/harness/test_walking_skeleton.py tests/security/ -v
```

To run the full TESTING.md-canonical command (using unittest runner):

```bash
python -m unittest eval/harness/test_walking_skeleton.py tests/unit/test_llm_client_hardening.py tests/unit/test_router_calibration.py tests/unit/test_hardened_web.py tests/unit/test_social_adapters.py tests/unit/test_desktop_control.py tests/unit/test_execution_engine.py tests/security/test_permission_adversarial.py tests/unit/test_auto_mode.py tests/unit/test_terminal_subsystem.py tests/unit/test_voice_pipeline.py tests/unit/test_voice_bugfix.py tests/unit/test_voice_web_api.py tests/unit/test_rich_response.py tests/unit/test_longform_writer.py
```
