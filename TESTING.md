# TESTING & VERIFICATION METHODOLOGY

**Project**: Antigravity Local Bounded-Autonomy Computer-Operating Agent (v3)  
**Status**: [VERIFIED] 59 Tests Passing Cleanly  

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

## 4. Test Suite Execution Command

To run the complete verified test suite across all subsystems:

```bash
python -m unittest eval/harness/test_walking_skeleton.py tests/unit/test_llm_client_hardening.py tests/unit/test_router_calibration.py tests/unit/test_hardened_web.py tests/unit/test_social_adapters.py tests/unit/test_desktop_control.py tests/unit/test_execution_engine.py tests/security/test_permission_adversarial.py tests/unit/test_auto_mode.py tests/unit/test_terminal_subsystem.py tests/unit/test_voice_pipeline.py
```
