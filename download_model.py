"""Download and cache Qwen2.5-1.5B-Instruct base weights with real-time visual progress."""

import sys
import time
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    print("=" * 60)
    print(f"Downloading Base Model weights for: {MODEL_ID}")
    print("This base model is ~3.1 GB and is required to run your LoRA adapter.")
    print("Once downloaded, it is permanently cached and never downloaded again.")
    print("=" * 60)
    print()

    t0 = time.time()
    try:
        path = snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=["*.safetensors", "*.json", "*.txt"],
            resume_download=True,
        )
        elapsed = time.time() - t0
        print()
        print("=" * 60)
        print(f" SUCCESS! Model downloaded and cached in {elapsed:.1f} seconds.")
        print(f"Location: {path}")
        print("=" * 60)
    except Exception as e:
        print(f"\n Error during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
