"""Robust resume downloader for Qwen2.5-1.5B-Instruct model weights with auto-reconnect."""

import os
import sys
import shutil
import time
import requests

URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors"
EXPECTED_SIZE = 3087467144
BLOB_HASH = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"

HOME = os.path.expanduser("~")
HUB_DIR = os.path.join(HOME, ".cache", "huggingface", "hub", "models--Qwen--Qwen2.5-1.5B-Instruct")
BLOBS_DIR = os.path.join(HUB_DIR, "blobs")
SNAPSHOT_DIR = os.path.join(HUB_DIR, "snapshots", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")

def main():
    os.makedirs(BLOBS_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    final_blob_path = os.path.join(BLOBS_DIR, BLOB_HASH)
    snapshot_model_path = os.path.join(SNAPSHOT_DIR, "model.safetensors")

    if os.path.exists(final_blob_path) and os.path.getsize(final_blob_path) == EXPECTED_SIZE:
        print("Base model weights already downloaded and complete!", flush=True)
        if not os.path.exists(snapshot_model_path):
            shutil.copy2(final_blob_path, snapshot_model_path)
        return

    # Find the largest incomplete file to resume
    candidates = []
    for fname in os.listdir(BLOBS_DIR):
        if fname.startswith(BLOB_HASH) and fname.endswith(".incomplete"):
            fpath = os.path.join(BLOBS_DIR, fname)
            candidates.append((os.path.getsize(fpath), fpath))

    candidates.sort(reverse=True)
    target_incomplete = candidates[0][1] if candidates else os.path.join(BLOBS_DIR, f"{BLOB_HASH}.part.incomplete")

    print(f"Target incomplete file: {target_incomplete}", flush=True)

    max_retries = 20
    retry_count = 0

    while retry_count < max_retries:
        current_size = os.path.getsize(target_incomplete) if os.path.exists(target_incomplete) else 0

        if current_size >= EXPECTED_SIZE:
            print("Download 100% complete!", flush=True)
            break

        headers = {"Range": f"bytes={current_size}-"} if current_size > 0 else {}
        mode = "ab" if current_size > 0 else "wb"
        percent = (current_size / EXPECTED_SIZE) * 100
        remaining_mb = (EXPECTED_SIZE - current_size) / (1024 * 1024)

        print(f"\n[Attempt {retry_count+1}] Resuming from {current_size / (1024**3):.2f} GB ({percent:.1f}%) | {remaining_mb:.1f} MB remaining...", flush=True)

        try:
            t0 = time.time()
            last_print = t0
            downloaded_session = 0

            with requests.get(URL, headers=headers, stream=True, allow_redirects=True, timeout=20) as r:
                if r.status_code not in (200, 206):
                    print(f"Server responded with status {r.status_code}. Retrying...", flush=True)
                    time.sleep(3)
                    retry_count += 1
                    continue

                with open(target_incomplete, mode) as f:
                    for chunk in r.iter_content(chunk_size=512 * 1024):  # 512 KB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded_session += len(chunk)
                            now = time.time()
                            if now - last_print >= 3.0:
                                total_downloaded = current_size + downloaded_session
                                pct = (total_downloaded / EXPECTED_SIZE) * 100
                                speed = (downloaded_session / max(0.1, now - t0)) / (1024 * 1024)
                                rem_mb = (EXPECTED_SIZE - total_downloaded) / (1024 * 1024)
                                eta = (rem_mb / speed) if speed > 0 else 0
                                print(f"Progress: {total_downloaded / (1024**3):.2f}/{EXPECTED_SIZE / (1024**3):.2f} GB ({pct:.1f}%) | Speed: {speed:.1f} MB/s | ETA: {eta:.0f}s", flush=True)
                                last_print = now

        except (requests.exceptions.RequestException, Exception) as e:
            print(f"Network hiccup: {e}. Reconnecting in 3s...", flush=True)
            time.sleep(3)
            retry_count += 1

    final_size = os.path.getsize(target_incomplete) if os.path.exists(target_incomplete) else 0
    if final_size == EXPECTED_SIZE:
        print("\n========================================================", flush=True)
        print(" DOWNLOAD COMPLETE! Assembling cache files...", flush=True)
        print("========================================================", flush=True)
        if os.path.exists(final_blob_path):
            os.remove(final_blob_path)
        shutil.move(target_incomplete, final_blob_path)
        print("Copied to HuggingFace snapshot directory...", flush=True)
        shutil.copy2(final_blob_path, snapshot_model_path)
        print(" SUCCESS: Base model is fully cached on disk!", flush=True)
    else:
        print(f"Finished with size {final_size} / {EXPECTED_SIZE}", flush=True)

if __name__ == "__main__":
    main()
