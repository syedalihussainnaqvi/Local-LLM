"""Interactive CLI interface to chat with the fine-tuned LoRA model and web-scraping RAG system."""

import os
import sys
import threading
import torch

from . import config
from .llm_engine import get_llm_engine
from .rag_pipeline import get_rag_pipeline
from .storage import get_storage


def print_banner(mode: str, web_enabled: bool):
    print("=" * 65)
    print("   Qwen2.5-1.5B-Instruct + LoRA Adapter + Web RAG Console")
    print("=" * 65)
    print(f"Base Model    : {config.BASE_MODEL_NAME}")
    print(f"Adapter Dir   : {config.ADAPTER_DIR}")
    device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    print(f"Compute       : {device}")
    print(f"Search Mode   : {mode.upper()}")
    print(f"Web Access    : {'ENABLED' if web_enabled else 'DISABLED'}")
    print("-" * 65)
    print("Commands:")
    print("  exit / quit     : Exit the program")
    print("  /clear          : Clear chat history")
    print("  /system <msg>   : Change system prompt")
    print("  /scrape <url>   : Scrape and add a webpage to Knowledge Base")
    print("  /sources        : List all stored knowledge base sources")
    print("  /refresh [url]  : Incremental refresh of sources")
    print("  /delete <url>   : Remove a source from the knowledge base")
    print("  /mode <m>       : Change mode (auto | local | live | llm)")
    print("  /web <on|off>   : Enable or disable live web access")
    print("=" * 65)
    print()


def main():
    mode = "auto"
    web_enabled = config.WEB_ACCESS_ENABLED

    print_banner(mode, web_enabled)
    print("Initializing LLM engine & loading weights (please wait)...")
    engine = get_llm_engine()
    engine.load_model()
    print("Model loaded successfully! You can start prompting.\n")

    pipeline = get_rag_pipeline()
    storage = get_storage()

    system_prompt = "You are a helpful, respectful, and honest assistant."
    history = []

    while True:
        try:
            prompt = input(f"\nYou [{mode}]> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", ":q"):
            print("Exiting. Goodbye!")
            break
        if prompt.lower() == "/clear":
            history = []
            print("[History cleared]")
            continue
        if prompt.lower().startswith("/system "):
            system_prompt = prompt[8:].strip()
            print(f"[System prompt set to: {system_prompt}]")
            continue
        if prompt.lower().startswith("/scrape "):
            target_url = prompt[8:].strip()
            print(f"[Scraping: {target_url} ...]")
            res = pipeline.scrape_and_store_url(target_url)
            if res.get("status") == "success":
                print(f"[SUCCESS] Added '{res.get('title')}' with {res.get('chunk_count')} chunks.")
            else:
                print(f"[ERROR] Failed to scrape: {res.get('error')}")
            continue
        if prompt.lower() == "/sources":
            sources = storage.get_all_sources()
            print(f"\nStored Sources ({len(sources)}):")
            for idx, s in enumerate(sources, 1):
                status_icon = "✓" if s["status"] == "active" else "✗"
                print(f" {idx}. [{status_icon}] {s['title']}")
                print(f"     URL     : {s['url']}")
                print(f"     Chunks  : {s.get('chunk_count', 0)} | Version: {s['version']} | Updated: {s['retrieved_at']}")
            print()
            continue
        if prompt.lower().startswith("/refresh"):
            arg = prompt[8:].strip()
            target_url = arg if arg else None
            print(f"[Refreshing {'all sources' if not target_url else target_url} ...]")
            res = pipeline.refresh_sources(target_url=target_url)
            print(f"[Refresh Summary] Updated: {res['updated']}, Skipped: {res['skipped']}, Errors: {res['errors']}")
            continue
        if prompt.lower().startswith("/delete "):
            del_url = prompt[8:].strip()
            if storage.delete_source(del_url):
                pipeline.retriever.refresh_index()
                print(f"[DELETED] Source '{del_url}' removed.")
            else:
                print(f"[ERROR] Source '{del_url}' not found.")
            continue
        if prompt.lower().startswith("/mode "):
            new_mode = prompt[6:].strip().lower()
            if new_mode in ("auto", "local", "live", "llm"):
                mode = "local_rag" if new_mode == "local" else ("live_web" if new_mode == "live" else ("llm_only" if new_mode == "llm" else "auto"))
                print(f"[Search mode set to: {mode.upper()}]")
            else:
                print("Invalid mode. Choose from: auto, local, live, llm")
            continue
        if prompt.lower().startswith("/web "):
            choice = prompt[5:].strip().lower()
            web_enabled = choice in ("on", "true", "1", "yes")
            print(f"[Web access set to: {'ENABLED' if web_enabled else 'DISABLED'}]")
            continue

        # Execute RAG orchestration
        print("[Processing query...]", flush=True)
        rag_res = pipeline.prepare_rag_request(
            prompt=prompt,
            system_prompt=system_prompt,
            mode=mode,
            web_access_enabled=web_enabled,
            status_callback=lambda s: print(f"  -> {s}"),
        )

        # Format history payload
        history_payload = []
        for u_msg, a_msg in history:
            history_payload.append({"user": u_msg, "assistant": a_msg})

        stream_gen = engine.generate_stream(
            question=rag_res["augmented_prompt"],
            context=None,
            history=history_payload,
            system_prompt=rag_res["effective_system_prompt"],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )

        print("\nModel > ", end="", flush=True)
        response_text = ""
        for item in stream_gen:
            if "token" in item:
                token = item["token"]
                print(token, end="", flush=True)
                response_text += token

        print()

        # Display citations if any were used
        citations = rag_res.get("citations", [])
        if citations:
            print("\n" + "-" * 50)
            print("Sources Cited:")
            for c in citations:
                print(f" • {c['title']}: {c['url']} (Score: {c.get('score', 0):.2f})")
            print("-" * 50)

        # Handle Desktop Action Approval directly in CLI
        if rag_res.get("route") == "desktop_action":
            from .desktop_agent import get_desktop_agent
            agent = get_desktop_agent()
            pending = agent.get_pending_plan()
            if pending:
                print("\n" + "=" * 55)
                print(f"⚠️  AUTONOMOUS DESKTOP CONTROL: Plan {pending.plan_id}")
                print(f"Risk Level : {pending.risk_summary}")
                print("=" * 55)
                choice = input("\nDo you grant permission to execute this plan on your computer? [y/N]: ").strip().lower()
                if choice in ("y", "yes"):
                    print(f"\n[Executing Plan {pending.plan_id} on your computer...]")
                    agent.grant_permission(pending.plan_id)
                    for ev in agent.execute_plan(pending.plan_id, auto_approve=True):
                        if ev.get("type") == "status":
                            print(f"  -> {ev.get('message', '')}")
                        elif ev.get("type") == "plan_complete":
                            print(f"\n[DONE] {ev.get('message', '')}")
                else:
                    agent.reject_permission(pending.plan_id)
                    print(f"\n[CANCELLED] Plan {pending.plan_id} was rejected. No actions performed.")

        history.append((prompt, response_text.strip()))


if __name__ == "__main__":
    main()
