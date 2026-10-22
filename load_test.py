import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"[{time.strftime('%X')}] Step 1: Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
print(f"[{time.strftime('%X')}] Tokenizer loaded.")

print(f"[{time.strftime('%X')}] Step 2: Downloading & Loading Base Model '{BASE_MODEL_NAME}'...")
t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
print(f"[{time.strftime('%X')}] Base model loaded in {time.time()-t0:.1f}s.")

print(f"[{time.strftime('%X')}] Step 3: Applying LoRA Adapter from '{ADAPTER_DIR}'...")
t1 = time.time()
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()
print(f"[{time.strftime('%X')}] LoRA adapter applied in {time.time()-t1:.1f}s.")

print(f"[{time.strftime('%X')}] Step 4: Testing prompt generation...")
prompt = "Hello! Who are you?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=32)
reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"[{time.strftime('%X')}] Model output: {reply}")
print("ALL TESTS PASSED!")
