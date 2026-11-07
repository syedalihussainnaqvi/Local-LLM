import json
import logging
import os
import shutil
import sys
import threading
import time

from flask import Flask, Response, jsonify, render_template, request

from . import config
from .llm_engine import (
    CLAUDE_SYSTEM_PROMPT,
    PROMPT_A_MARKDOWN,
    PROMPT_A_PREFILL,
    PROMPT_A_SUFFIX,
    PROMPT_B_HTML,
    PROMPT_B_PREFILL,
    PROMPT_B_SUFFIX,
    RICH_FORMAT_SUFFIX,
    RICH_RESPONSE_SHORT_PROMPT,
    RICH_RESPONSE_SYSTEM_PROMPT,
    FAST_VISUAL_PREFILL,
    FAST_VISUAL_SUFFIX,
    FAST_VISUAL_SYSTEM_PROMPT,
    CODE_IMAGE_PREFILL,
    CODE_IMAGE_SUFFIX,
    CODE_IMAGE_SYSTEM_PROMPT,
    CodeImageValidator,
    is_image_or_diagram_request,
    build_fast_visual_user_message,
    build_page,
    build_user_message,
    extract_topic_and_purpose,
    get_llm_engine,
)
from .rag_pipeline import get_rag_pipeline
from .storage import get_storage
from .trending_engine import get_trending_engine
from .social_scraper import get_social_scraper
from .verification_engine import get_verification_engine
from .script_engine import get_script_engine
from .image_router import wants_image, route_image_request
from .desktop_agent import get_desktop_agent

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("app")

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Singletons
engine = get_llm_engine()
pipeline = get_rag_pipeline()
storage = get_storage()
trending_engine = get_trending_engine()
social_scraper = get_social_scraper()
verification_engine = get_verification_engine()
script_engine = get_script_engine()
desktop_agent = get_desktop_agent()

# Voice Pipeline (Section 6A)
from .agent.voice.pipeline import VoicePipeline
voice_pipeline = VoicePipeline()


def load_model_worker():
    try:
        engine.load_model()
    except Exception as e:
        logger.error(f"Error loading model: {e}")


@app.route("/api/status", methods=["GET"])
def get_status():
    """Return model status and RAG knowledge base statistics."""
    status = engine.get_status()
    stats = storage.get_stats()
    status["stats"] = stats
    status["web_access_enabled"] = config.WEB_ACCESS_ENABLED
    status["search_provider"] = config.SEARCH_PROVIDER
    return jsonify(status)


@app.route("/api/load", methods=["POST"])
def trigger_load():
    """Trigger background loading of base model and LoRA adapter."""
    if not engine.is_loaded and not engine.is_loading:
        thread = threading.Thread(target=load_model_worker, daemon=True)
        thread.start()
    return jsonify(engine.get_status())


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Streaming chat endpoint with integrated RAG pipeline.
    Accepts:
        prompt: str (required)
        system_prompt: str (optional)
        history: list of turns (optional)
        max_tokens: int (default 512)
        temperature: float (default 0.7)
        top_p: float (default 0.9)
        search_mode: "auto" | "local_rag" | "live_web" | "llm_only" (default "auto")
        web_access_enabled: bool (default config.WEB_ACCESS_ENABLED)
    """
    if not engine.is_loaded:
        return jsonify({"error": "Model is not loaded yet. Please wait."}), 400

    data = request.get_json() or {}
    user_prompt = data.get("prompt", "").strip()
    system_prompt = data.get("system_prompt", "").strip() or CLAUDE_SYSTEM_PROMPT
    history = data.get("history", [])
    max_tokens = min(int(data.get("max_tokens", 1024)), 2048)
    temperature = float(data.get("temperature", 0.7))
    top_p = float(data.get("top_p", 0.9))
    search_mode = data.get("search_mode", "auto")
    web_access_enabled = data.get("web_access_enabled", config.WEB_ACCESS_ENABLED)
    prefill = data.get("prefill", None)

    if not user_prompt:
        return jsonify({"error": "Prompt cannot be empty"}), 400

    # Format enforcement for visual images/diagrams, notes/study/quiz requests
    lower_prompt = user_prompt.lower()
    is_code_image = (
        "CAN create images, logos, icons" in system_prompt
        or system_prompt == CODE_IMAGE_SYSTEM_PROMPT
        or data.get("format_preset") == "code_image"
        or (is_image_or_diagram_request(user_prompt) and not any(k in lower_prompt for k in ["quiz", "test me", "flashcard", "mcq"]))
    )
    is_fast_visual = (
        "Output ONLY one valid JSON object" in system_prompt
        or system_prompt == FAST_VISUAL_SYSTEM_PROMPT
        or data.get("format_preset") == "fast_visual"
    )
    is_html_mode = "study-notes engine" in system_prompt or "const DATA=" in system_prompt
    is_study_or_quiz = any(k in lower_prompt for k in ["quiz", "detailed note", "study note", "test me", "flashcard", "mcq", "exam prep"])

    if is_code_image:
        if not user_prompt.endswith(CODE_IMAGE_SUFFIX):
            user_prompt += CODE_IMAGE_SUFFIX
        if not system_prompt or system_prompt == CLAUDE_SYSTEM_PROMPT:
            system_prompt = CODE_IMAGE_SYSTEM_PROMPT
        if prefill is None:
            prefill = CODE_IMAGE_PREFILL
        temperature = min(temperature, 0.2)
        max_tokens = min(max_tokens, 1800)
    elif is_fast_visual:
        if "TOPIC:" not in user_prompt.upper():
            topic, purpose = extract_topic_and_purpose(user_prompt)
            user_prompt = build_fast_visual_user_message(topic=topic, purpose=purpose)
        if not user_prompt.endswith(FAST_VISUAL_SUFFIX):
            user_prompt += FAST_VISUAL_SUFFIX
        if prefill is None:
            prefill = FAST_VISUAL_PREFILL
        temperature = min(temperature, 0.4)
        max_tokens = min(max_tokens, 1500)
    elif is_html_mode:
        if not user_prompt.endswith(PROMPT_B_SUFFIX):
            user_prompt += PROMPT_B_SUFFIX
        if prefill is None:
            prefill = PROMPT_B_PREFILL
    elif is_study_or_quiz:
        # Part 2 Topic Lock: Structure user prompt to lock subject and prevent topic drift
        if "TOPIC:" not in user_prompt.upper():
            topic, purpose = extract_topic_and_purpose(user_prompt)
            user_prompt = build_user_message(topic=topic, purpose=purpose)
        if not (user_prompt.endswith(PROMPT_A_SUFFIX) or user_prompt.endswith(RICH_FORMAT_SUFFIX)):
            user_prompt += PROMPT_A_SUFFIX
        if not system_prompt or system_prompt == CLAUDE_SYSTEM_PROMPT:
            system_prompt = PROMPT_A_MARKDOWN
        if prefill is None:
            prefill = PROMPT_A_PREFILL
        temperature = min(temperature, 0.5)

    def generate_events():
        # ── IMAGE FAST-PATH (4-layer image router) ────────────────────────────
        # If the user is asking for an image/diagram and the code_image preset
        # is NOT explicitly set (which has its own dedicated flow below), we
        # intercept here so the refusal can never reach the user.
        _is_explicit_code_image = (
            is_code_image and data.get("format_preset") == "code_image"
        )
        if wants_image(user_prompt) and not _is_explicit_code_image:
            try:
                result = route_image_request(engine, user_prompt)
                html_code = result["html"]
                # Stream the HTML as a single token then done
                yield f"data: {json.dumps({'token': html_code})}\n\n"
                yield f"data: {json.dumps({'done': True, 'reply': html_code, 'token_count': 1, 'duration': 0, 'speed': 0})}\n\n"
                return
            except Exception as img_exc:
                logger.warning("image_router failed, falling through to normal pipeline: %s", img_exc)

        # Step 1: Execute RAG orchestration with real-time streaming progress events
        try:
            rag_result = None
            for ev in pipeline.stream_rag_workflow(
                prompt=user_prompt,
                system_prompt=system_prompt,
                mode=search_mode,
                web_access_enabled=web_access_enabled,
            ):
                if ev["type"] == "status":
                    yield f"data: {json.dumps({'rag_status': ev['message'], 'action': ev.get('action'), 'url': ev.get('url'), 'title': ev.get('title'), 'source': ev.get('source'), 'step_id': ev.get('step_id')})}\n\n"
                elif ev["type"] == "desktop_plan":
                    yield f"data: {json.dumps({'desktop_plan': ev['plan'], 'plan_message': ev.get('message')})}\n\n"
                elif ev["type"] == "plan_complete":
                    yield f"data: {json.dumps({'plan_complete': ev['plan'], 'plan_message': ev.get('message')})}\n\n"
                elif ev["type"] == "rag_complete":
                    rag_result = ev["result"]

            if not rag_result:
                rag_result = pipeline.prepare_rag_request(
                    user_prompt, system_prompt, search_mode, web_access_enabled
                )

            citations = rag_result.get("citations", [])
            route = rag_result.get("route", "llm_only")

            # Emit routing and citations event before streaming tokens
            yield f"data: {json.dumps({'route': route, 'citations': citations})}\n\n"

            # ── Fast-path for Desktop Actions (Instant response, NO slow multi-minute LLM loop) ──
            if route == "desktop_action":
                fast_notice = "Action plan generated. Please click **Allow & Execute** or **Decline** above or in the popup to proceed."
                yield f"data: {json.dumps({'token': fast_notice})}\n\n"
                yield f"data: {json.dumps({'done': True, 'citations': citations, 'route': route})}\n\n"
                return

            if route == "desktop_action_reject":
                cancel_notice = "Action plan was cancelled. No commands or actions were executed on your laptop."
                yield f"data: {json.dumps({'token': cancel_notice})}\n\n"
                yield f"data: {json.dumps({'done': True, 'citations': citations, 'route': route})}\n\n"
                return

            # Step 2: Stream tokens from local fine-tuned LLM
            # For grounded RAG queries, use a focused temperature to ensure exact numerical accuracy
            effective_temperature = min(temperature, 0.15) if route in ("live_web", "local_rag", "direct_url") else temperature

            # For live web factual lookups, avoid stale prior assistant turns poisoning fresh numerical context
            effective_history = None if route in ("live_web", "direct_url") else history

            stream_gen = engine.generate_stream(
                question=rag_result["augmented_prompt"],
                context=None,  # Already seamlessly integrated into augmented_prompt
                history=effective_history,
                system_prompt=rag_result["effective_system_prompt"],
                max_tokens=max_tokens,
                temperature=effective_temperature,
                top_p=top_p,
                prefill=prefill,
            )

            for item in stream_gen:
                if "token" in item:
                    yield f"data: {json.dumps({'token': item['token']})}\n\n"
                elif item.get("done"):
                    # Attach citations to the final stats payload
                    item["citations"] = citations
                    item["route"] = route
                    yield f"data: {json.dumps(item)}\n\n"

        except Exception as err:
            logger.error(f"Inference error: {err}", exc_info=True)
            yield f"data: {json.dumps({'error': str(err)})}\n\n"

    return Response(generate_events(), mimetype="text/event-stream")


@app.route("/api/scrape", methods=["POST"])
def scrape_endpoint():
    """Manually scrape a URL and add it to the local knowledge base."""
    data = request.get_json() or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL parameter is required"}), 400

    try:
        res = pipeline.scrape_and_store_url(url)
        if res.get("status") != "success":
            return jsonify(res), 400
        return jsonify(res)
    except Exception as e:
        logger.error(f"Scraping endpoint failed for {url}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/trending", methods=["GET"])
def get_trending_endpoint():
    """Fetch real-time trending topics across Google Trends, TikTok, X, YouTube, Reddit, and GitHub."""
    category = request.args.get("category", "all")
    limit = int(request.args.get("limit", 20))
    try:
        trends = trending_engine.get_all_trending_topics(category=category, limit=limit)
        return jsonify({
            "status": "success",
            "category": category,
            "count": len(trends),
            "trends": trends,
        })
    except Exception as e:
        logger.error(f"Error fetching trending topics: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/social-search", methods=["POST"])
def social_search_endpoint():
    """Search and scrape across social media platforms or internet-wide."""
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    platforms = data.get("platforms", None)
    limit = int(data.get("limit", 4))
    all_internet = data.get("all_internet", False)

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    try:
        if all_internet:
            results = social_scraper.search_internet_wide(query, max_results=limit * 3)
            return jsonify({"status": "success", "results": results, "count": len(results)})
        res = social_scraper.search_all_social_media(query, platforms=platforms, limit_per_platform=limit)
        return jsonify({"status": "success", **res})
    except Exception as e:
        logger.error(f"Social search endpoint error for '{query}': {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/script", methods=["POST"])
def generate_script_endpoint():
    """Build a viral script prompt context with real-time trending hooks, news, and platform templates."""
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt parameter is required"}), 400
    try:
        script_res = script_engine.build_script_prompt_context(prompt)
        return jsonify({"status": "success", **script_res})
    except Exception as e:
        logger.error(f"Script endpoint error for '{prompt}': {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/scrape", methods=["POST"])
def scrape_url_endpoint():
    """Deep scrape any website or social media URL on the internet."""
    data = request.get_json() or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL parameter is required"}), 400
    try:
        scraped = social_scraper.scrape_any_website(url)
        return jsonify({"status": "success", "data": scraped})
    except Exception as e:
        logger.error(f"Scrape endpoint error for '{url}': {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/verify", methods=["POST"])
def verify_claim_endpoint():
    """Cross-verify a news claim or rumor across major news outlets, social platforms, and fact-checkers."""
    data = request.get_json() or {}
    claim = data.get("claim", "").strip()

    if not claim:
        return jsonify({"error": "Claim parameter is required"}), 400

    try:
        report = verification_engine.verify_claim(claim)
        return jsonify({"status": "success", "report": report})
    except Exception as e:
        logger.error(f"Verification endpoint error for '{claim}': {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/desktop/plan", methods=["POST"])
def desktop_plan_endpoint():
    """Create an autonomous desktop action plan from user prompt."""
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt parameter is required"}), 400
    try:
        from dataclasses import asdict
        plan = desktop_agent.plan_task(prompt)
        return jsonify({"status": "success", "plan": asdict(plan)})
    except Exception as e:
        logger.error(f"Desktop plan error: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/desktop/permission", methods=["POST"])
def desktop_permission_endpoint():
    """Grant or reject permission to execute an action plan on the user's computer."""
    data = request.get_json() or {}
    plan_id = data.get("plan_id", "").strip()
    decision = data.get("decision", "approve").lower()
    if not plan_id:
        return jsonify({"error": "Plan ID is required"}), 400

    if decision in ("approve", "grant", "yes"):
        res = desktop_agent.grant_permission(plan_id)
    else:
        res = desktop_agent.reject_permission(plan_id)
    return jsonify(res)


@app.route("/api/desktop/os-popup", methods=["POST"])
def desktop_os_popup_endpoint():
    """Trigger a native Windows OS top-most modal popup dialog with Allow/Decline buttons."""
    data = request.get_json() or {}
    plan_id = data.get("plan_id", "").strip()
    if not plan_id:
        pending = desktop_agent.get_pending_plan()
        if pending:
            plan_id = pending.plan_id
        else:
            return jsonify({"error": "Plan ID is required or no pending plan found"}), 400

    allowed = desktop_agent.prompt_windows_native_popup(plan_id)
    return jsonify({
        "status": "success",
        "plan_id": plan_id,
        "decision": "approved" if allowed else "rejected",
        "allowed": allowed,
    })


@app.route("/api/desktop/pending-plan", methods=["GET"])
def desktop_pending_plan_endpoint():
    """Retrieve the active pending desktop action plan awaiting user permission."""
    plan = desktop_agent.get_pending_plan()
    if plan:
        from dataclasses import asdict
        return jsonify({"status": "success", "has_pending": True, "plan": asdict(plan)})
    return jsonify({"status": "success", "has_pending": False, "plan": None})


@app.route("/api/desktop/execute", methods=["POST"])
def desktop_execute_endpoint():
    """Stream execution of an authorized desktop action plan."""
    data = request.get_json() or {}
    plan_id = data.get("plan_id", "").strip()
    auto_approve = bool(data.get("auto_approve", False))
    if not plan_id:
        return jsonify({"error": "Plan ID is required"}), 400

    def generate():
        for event in desktop_agent.execute_plan(plan_id, auto_approve=auto_approve):
            yield f"data: {json.dumps(event)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/desktop/action", methods=["POST"])
def desktop_direct_action_endpoint():
    """Execute a single direct desktop action with permission check."""
    data = request.get_json() or {}
    action = data.get("action", "").strip()
    params = data.get("params", {})
    permission_granted = bool(data.get("permission_granted", False))

    if not permission_granted:
        return jsonify({
            "status": "permission_required",
            "message": "Explicit user permission is required to perform actions on this laptop.",
            "action": action,
            "params": params,
        }), 403

    executor = desktop_agent.executor
    try:
        if action == "open_app":
            res = executor.open_app(params.get("app_name", ""), params.get("path"), params.get("args"))
        elif action == "close_app":
            res = executor.close_app(params.get("app_name", ""))
        elif action == "whatsapp_send":
            res = executor.whatsapp_send_message(params.get("recipient", ""), params.get("message", ""))
        elif action == "whatsapp_delete":
            res = executor.whatsapp_delete_message()
        elif action == "browser_search":
            res = executor.browser_search(params.get("query", ""))
        elif action == "browser_scroll":
            res = executor.browser_scroll(params.get("direction", "down"), params.get("amount", 600))
        elif action == "browser_chatgpt":
            res = executor.browser_chatgpt_prompt(params.get("task_prompt", ""))
        elif action == "run_terminal":
            res = executor.run_terminal_command(params.get("command", ""))
        elif action == "create_file":
            res = executor.create_file(params.get("file_path", ""), params.get("content", ""))
        elif action == "edit_file":
            res = executor.edit_file(params.get("file_path", ""), params.get("content", ""), params.get("mode", "overwrite"))
        elif action == "delete_file":
            res = executor.delete_file(params.get("file_path", ""))
        elif action == "mouse_click":
            res = executor.mouse_click(params.get("x"), params.get("y"), params.get("button", "left"), params.get("clicks", 1))
        elif action == "keyboard_type":
            res = executor.keyboard_type(params.get("text", ""), params.get("enter", False))
        elif action == "screenshot":
            res = executor.capture_screenshot()
        else:
            return jsonify({"status": "error", "error": f"Unknown action: {action}"}), 400
        return jsonify({"status": "success", "result": res})
    except Exception as e:
        logger.error(f"Desktop action error ({action}): {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/sources", methods=["GET"])
def get_sources_endpoint():
    """List all stored knowledge base sources and global statistics."""
    sources = storage.get_all_sources()
    stats = storage.get_stats()
    return jsonify({"sources": sources, "stats": stats})


@app.route("/api/sources", methods=["DELETE"])
def delete_source_endpoint():
    """Remove a source and its associated chunks from local storage."""
    data = request.get_json() or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL parameter is required"}), 400

    success = storage.delete_source(url)
    if success:
        pipeline.retriever.refresh_index()
        return jsonify({"status": "success", "message": f"Source '{url}' and chunks deleted."})
    return jsonify({"status": "error", "message": "Source not found"}), 404


@app.route("/api/sources/refresh", methods=["POST"])
def refresh_sources_endpoint():
    """Refresh all stored sources or a specific URL."""
    data = request.get_json() or {}
    target_url = data.get("url", "").strip() or None
    summary = pipeline.refresh_sources(target_url=target_url)
    return jsonify({"status": "success", "summary": summary})


@app.route("/api/history", methods=["GET"])
def get_history_endpoint():
    """Retrieve audit crawl history logs."""
    limit = int(request.args.get("limit", 50))
    history = storage.get_history(limit=limit)
    return jsonify({"history": history})


@app.route("/api/config", methods=["GET", "POST"])
def manage_config():
    """Get or update runtime configuration options."""
    if request.method == "POST":
        data = request.get_json() or {}
        if "web_access_enabled" in data:
            config.WEB_ACCESS_ENABLED = bool(data["web_access_enabled"])
        if "search_provider" in data:
            config.SEARCH_PROVIDER = str(data["search_provider"])
        if "max_urls_per_query" in data:
            config.MAX_URLS_PER_QUERY = int(data["max_urls_per_query"])
        if "retrieval_top_k" in data:
            config.RETRIEVAL_TOP_K = int(data["retrieval_top_k"])
        return jsonify({"status": "success", "message": "Configuration updated"})

    return jsonify({
        "web_access_enabled": config.WEB_ACCESS_ENABLED,
        "search_provider": config.SEARCH_PROVIDER,
        "max_urls_per_query": config.MAX_URLS_PER_QUERY,
        "retrieval_top_k": config.RETRIEVAL_TOP_K,
        "request_timeout": config.REQUEST_TIMEOUT,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
    })



@app.route("/api/voice/stt", methods=["POST"])
def voice_stt_offline():
    """Offline STT endpoint: accepts a WAV/webm audio blob, converts & transcribes
    using Windows SAPI System.Speech (100% local, no cloud). Returns transcript."""
    import tempfile, os, subprocess
    from .agent.voice.audio_engine import WindowsSTT

    audio_data = request.get_data()
    if not audio_data:
        return jsonify({"error": "No audio data received"}), 400

    tmp_dir = str(config.PROJECT_ROOT / "eval" / "fixtures")
    os.makedirs(tmp_dir, exist_ok=True)

    raw_path = os.path.join(tmp_dir, f"stt_upload_{int(time.time()*1000)}.bin")
    wav_path = os.path.join(tmp_dir, f"stt_converted_{int(time.time()*1000)}.wav")

    try:
        with open(raw_path, "wb") as f:
            f.write(audio_data)

        if audio_data.startswith(b"RIFF"):
            shutil.copy(raw_path, wav_path)
        else:
            ffmpeg_available = shutil.which("ffmpeg") is not None
            if ffmpeg_available:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", raw_path,
                     "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                    capture_output=True, timeout=15
                )
            else:
                shutil.copy(raw_path, wav_path)

        result = WindowsSTT.transcribe_wav(wav_path, timeout_seconds=15.0)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Offline STT error: {e}", exc_info=True)
        return jsonify({"transcript": "", "error": str(e)}), 500
    finally:
        for p in [raw_path, wav_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


@app.route("/api/voice", methods=["POST"])

def voice_utterance():
    """Real-time voice turn endpoint bridging browser and VoicePipeline."""
    data = request.get_json() or {}
    transcript = data.get("transcript", "").strip()
    history = data.get("history", [])

    if not transcript:
        return jsonify({"error": "Transcript cannot be empty"}), 400

    def llm_gen(user_query: str) -> str:
        if not engine.is_loaded:
            return "I am running locally, but the model weights are currently loading. Please give me a moment."
        try:
            voice_system_prompt = (
                "You are Claude participating in a real-time voice call. "
                "Keep responses concise, natural, and conversational (1 to 3 sentences max). "
                "Do not use markdown formatting, code fences, or bullet lists."
            )
            res = engine.generate_answer(
                question=user_query,
                history=history[-4:] if history else None,
                system_prompt=voice_system_prompt,
                max_tokens=256,
                temperature=0.7,
            )
            return res.get("reply", "I heard you.")
        except Exception as e:
            logger.error(f"Voice LLM generation error: {e}", exc_info=True)
            return f"I encountered an issue generating a response: {e}"

    result = voice_pipeline.process_user_utterance(transcript, llm_generate_fn=llm_gen)
    return jsonify(result)


@app.route("/api/voice/control", methods=["POST"])
def voice_control():
    """Control fast-path and state toggles for voice session."""
    data = request.get_json() or {}
    action = data.get("action", "").lower().strip()

    if action == "barge_in":
        latency_ms = voice_pipeline.handle_barge_in()
        return jsonify({"status": "ok", "action": "barge_in", "latency_ms": latency_ms})
    elif action == "mute":
        muted = bool(data.get("muted", True))
        voice_pipeline.set_mute(muted)
        return jsonify({"status": "ok", "action": "mute", "is_muted": muted})
    elif action == "status":
        return jsonify({
            "status": "ok",
            "state": voice_pipeline.state.value,
            "is_muted": voice_pipeline.is_muted,
            "history_len": len(voice_pipeline.conversation_history),
        })
    return jsonify({"error": f"Unknown action: {action}"}), 400


@app.route("/api/presets", methods=["GET"])
def get_presets():
    """Serve system-prompt presets as JSON so the frontend never embeds raw strings in JS."""
    return jsonify({
        "prompt_a": PROMPT_A_MARKDOWN,
        "prompt_b": PROMPT_B_HTML,
        "fast_visual": FAST_VISUAL_SYSTEM_PROMPT,
        "fast_visual_prefill": FAST_VISUAL_PREFILL,
        "code_image": CODE_IMAGE_SYSTEM_PROMPT,
        "code_image_prefill": CODE_IMAGE_PREFILL,
        "claude_default": CLAUDE_SYSTEM_PROMPT,
    })


@app.route("/template.html", methods=["GET"])
def serve_template_html():
    """Serve the Fast Visual Study-Notes template HTML."""
    return render_template("template.html")


@app.route("/api/render-notes", methods=["POST"])
def render_notes_endpoint():
    """Compile JSON or model response into a standalone visual notes HTML page."""
    data = request.get_json() or {}
    model_text = data.get("model_text") or data.get("text") or ""
    if not model_text and "blocks" in data:
        model_text = json.dumps(data)
    if not model_text:
        return jsonify({"error": "No model_text or JSON blocks provided"}), 400

    try:
        html = build_page(model_text)
        return Response(html, mimetype="text/html")
    except Exception as e:
        logger.error(f"Error compiling visual notes page: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400


@app.route("/api/validate-image", methods=["POST"])
def validate_image_endpoint():
    """Validate model-generated image/diagram code against the 17 Error Prevention rules.

    Accepts: { "code": "...generated HTML..." }
    Returns: { "valid": true/false, "errors": [...], "repair_prompt": "..." }
    """
    req_data = request.get_json() or {}
    code_text = req_data.get("code", "").strip()
    if not code_text:
        return jsonify({"error": "No code provided"}), 400

    is_valid, errors, extracted = CodeImageValidator.validate(code_text)
    repair_prompt = (
        CodeImageValidator.build_repair_message(extracted, errors)
        if not is_valid else None
    )
    return jsonify({
        "valid": is_valid,
        "errors": errors,
        "repair_prompt": repair_prompt,
    })


@app.route("/")
def index():
    return render_template("index.html")


def main():
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Qwen 2.5 LoRA + Web RAG Server on http://127.0.0.1:{port}")
    # Trigger model background load right away
    thread = threading.Thread(target=load_model_worker, daemon=True)
    thread.start()
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
