"""Capability gate benchmark script for Antigravity Phase 0.
Tests:
1. Model loading latency & RAM footprint on Intel Core i5-8265U (CPU).
2. Token generation speed (tokens/sec) and time-to-first-token.
3. Tool-calling / JSON structured output fidelity.
4. Screen capture latency and coordinate resolution.
"""

import os
import sys
import time
import json
import psutil
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "src")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

def get_process_ram_mb():
    p = psutil.Process()
    return p.memory_info().rss / (1024 * 1024)

def benchmark():
    results = {}
    print("=== CAPABILITY GATE BENCHMARK ===")
    
    # 1. RAM Baseline
    ram_init = get_process_ram_mb()
    results["ram_initial_mb"] = round(ram_init, 1)
    print(f"Initial RAM usage: {ram_init:.1f} MB")
    
    # 2. Model & Tokenizer Load
    t0 = time.time()
    from local_llm.llm_engine import LLMEngine
    engine = LLMEngine()
    print("Loading model and LoRA adapter...")
    engine.load_model()
    t_load = time.time() - t0
    ram_after_load = get_process_ram_mb()
    results["model_load_seconds"] = round(t_load, 2)
    results["model_ram_mb"] = round(ram_after_load - ram_init, 1)
    results["ram_total_after_load_mb"] = round(ram_after_load, 1)
    print(f"Model loaded in {t_load:.2f}s | Additional RAM: {ram_after_load - ram_init:.1f} MB")
    
    # 3. Text Generation Speed Benchmark
    prompt = "What are the primary principles of operating system process isolation?"
    print("\nBenchmarking text generation (50 tokens)...")
    gen = engine.generate_stream(prompt, max_tokens=60, temperature=0.1)
    t_start = time.time()
    first_token_time = None
    token_count = 0
    full_text = ""
    for chunk in gen:
        if "token" in chunk:
            token_count += 1
            if first_token_time is None:
                first_token_time = time.time() - t_start
            full_text += chunk["token"]
        elif chunk.get("done"):
            stats = chunk
    total_gen_time = time.time() - t_start
    tps = token_count / total_gen_time if total_gen_time > 0 else 0
    
    results["ttft_seconds"] = round(first_token_time or 0, 3)
    results["tokens_generated"] = token_count
    results["gen_duration_seconds"] = round(total_gen_time, 2)
    results["tokens_per_second"] = round(tps, 2)
    print(f"TTFT: {results['ttft_seconds']}s | Tokens: {token_count} | Speed: {tps:.2f} tok/s")
    
    # 4. Tool Call / Structured JSON Prompting Test
    print("\nBenchmarking tool-call JSON emission...")
    tool_system_prompt = (
        "You are an OS agent. If you need to open an application, output ONLY a JSON object: "
        "{\"action\": \"open_app\", \"app\": \"<name>\"}."
    )
    t_tool_0 = time.time()
    ans = engine.generate_answer("Please open Google Chrome for me.", system_prompt=tool_system_prompt, max_tokens=40, temperature=0.01)
    t_tool = time.time() - t_tool_0
    reply = ans["reply"].strip()
    is_valid_json = False
    try:
        # check if json in reply
        parsed = None
        if "{" in reply and "}" in reply:
            start = reply.find("{")
            end = reply.rfind("}") + 1
            parsed = json.loads(reply[start:end])
            is_valid_json = isinstance(parsed, dict)
    except Exception:
        is_valid_json = False
        
    results["tool_json_valid"] = is_valid_json
    results["tool_raw_reply"] = reply
    results["tool_latency_seconds"] = round(t_tool, 2)
    print(f"Tool generation in {t_tool:.2f}s | Valid JSON: {is_valid_json}")
    print(f"Raw reply: {reply[:120]}")
    
    # 5. Screen Capture Latency
    print("\nBenchmarking screen capture...")
    import pyautogui
    t_cap_0 = time.time()
    shot = pyautogui.screenshot()
    t_cap = time.time() - t_cap_0
    results["screen_capture_seconds"] = round(t_cap, 3)
    results["screen_resolution"] = f"{shot.width}x{shot.height}"
    print(f"Screen captured in {t_cap:.3f}s | Resolution: {shot.width}x{shot.height}")
    
    # Save results
    os.makedirs("eval/reports", exist_ok=True)
    out_path = "eval/reports/capability_gate_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults successfully saved to {out_path}")
    return results

if __name__ == "__main__":
    benchmark()
