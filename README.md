---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct
- lora
- transformers
- rag
- web-scraper
---

# Local AI Chatbot with Custom Web-Scraping & RAG Integration

This repository hosts a local instruction-tuned LLM chatbot powered by **Qwen2.5-1.5B-Instruct** and a fine-tuned **PEFT LoRA adapter**, integrated with an enterprise-grade **Custom Web-Scraping & Retrieval-Augmented Generation (RAG)** system.

The system allows the model to retrieve information from websites, store and index documents in a persistent local SQLite knowledge base, answer user questions grounded in retrieved evidence, and provide accurate clickable source citations—**without needing to retrain the local model**.

---

## 1. System Architecture

```
                                  +-----------------------+
                                  |     User Interface    |
                                  |  (Web UI / CLI Chat)  |
                                  +-----------+-----------+
                                              |
                                     [User Prompt & Mode]
                                              |
                                              v
                                  +-----------+-----------+
                                  |      Query Router     |
                                  +-----+-----+-----+-----+
                                        |     |     |
                 +----------------------+     |     +----------------------+
                 | (Direct URL / Live Web)    | (Local KB Match)           | (General Q / LLM Only)
                 v                            v                            v
      +----------+----------+      +----------+----------+                 |
      |   Web Discovery &   |      |   Hybrid Retriever  |                 |
      |    Safe Scraper     |      |  (TF-IDF + Cosine)  |                 |
      +----------+----------+      +----------+----------+                 |
                 |                            |                            |
                 v                            |                            |
      +----------+----------+                 |                            |
      | Text Processor      |                 |                            |
      | (Sentence Chunking) |                 |                            |
      +----------+----------+                 |                            |
                 |                            |                            |
                 v                            |                            |
      +----------+----------+                 |                            |
      | SQLite Local Storage|<----------------+                            |
      | (rag_storage.db)    |                                              |
      +---------------------+                                              |
                 |                                                         |
                 +--------------------+                                    |
                                      | [Retrieved Context & Citations]    |
                                      v                                    |
                          +-----------+-----------+                        |
                          |  Prompt Synthesizer   |<-----------------------+
                          |  & Grounding Filter   |
                          +-----------+-----------+
                                      |
                                      v
                          +-----------+-----------+
                          | Existing Local AI     |
                          | (Qwen2.5-1.5B + LoRA) |
                          +-----------+-----------+
                                      |
                             [Streaming Answer &
                              Source Citations]
                                      |
                                      v
                          +-----------+-----------+
                          |   Response Delivery   |
                          +-----------------------+
```

### Core Components:
- **`src/local_llm/config.py`**: Centralized configuration management with safe defaults.
- **`src/local_llm/web_scraper.py`**: SSRF-protected web fetcher with redirect re-validation, response size caps, robots.txt support, and semantic HTML extraction.
- **`src/local_llm/text_processor.py`**: Whitespace normalizer and sentence-boundary-aware chunker with metadata tracking (source URL, title, section, hash).
- **`src/local_llm/storage.py`**: Persistent, transactional SQLite storage (`data/rag_storage.db`) supporting incremental updates, versioning, cascading deletions, and audit crawl history.
- **`src/local_llm/embeddings.py`**: Local, CPU-efficient hashing embedder producing normalized dense vectors without external downloads.
- **`src/local_llm/retrieval.py`**: Hybrid BM25 lexical + dense cosine retriever with threshold filtering and prompt context formatting.
- **`src/local_llm/query_router.py`**: Intent analyzer routing queries between `llm_only`, `local_rag`, `live_web`, and `direct_url`.
- **`src/local_llm/web_discovery.py`**: Discovery engine supporting DuckDuckGo, Brave Search API, and mock offline testing.
- **`src/local_llm/rag_pipeline.py`**: End-to-end orchestrator connecting routing, scraping, storage, retrieval, and grounding instructions.
- **`src/local_llm/llm_engine.py`**: Persistent, thread-safe inference manager for the Qwen2.5-1.5B base model and LoRA adapter.
- **`src/local_llm/app.py` & `src/local_llm/templates/index.html`**: Web UI with dark/light themes, live token streaming, Knowledge Base modal, and citation cards.
- **`src/local_llm/cli_chat.py`**: Interactive console interface supporting `/scrape`, `/sources`, `/refresh`, `/delete`, `/mode`, and `/web`.

## Project Layout

```text
src/local_llm/       Application package and agent integrations
  agent/             Agent core, tools, policy, and voice
  templates/         Web UI templates
models/adapter/      Fine-tuned adapter and tokenizer files
tests/               Automated tests
eval/                Evaluation code and fixtures
docs/                Architecture, security, testing, and operation guides
scripts/model/       Model download and load-test utilities
dev/scratch/         Experimental/manual development scripts
data/                Runtime database and local application data
```

The root `app.py`, `cli_chat.py`, and `agent_cli.py` files are small compatibility launchers. New application code belongs in `src/local_llm/`.

---

## 2. Prerequisites & Environment

- **Operating System**: Windows, Linux, or macOS.
- **Python**: Version 3.10 to 3.14.
- **Compute**:
  - **CPU**: Fully supported with `torch` CPU fallback (tested on PyTorch 2.14.1+cpu).
  - **GPU**: Automatically activated if CUDA is available.
- **RAM**: Minimum 8 GB recommended (model base footprint ~3.1 GB).

---

## 3. Installation Instructions

1. Clone or navigate to the project directory:
   ```powershell
   cd c:\Users\Jahan\Downloads\chat_model_final_adapter
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

4. Create your local environment configuration:
   ```powershell
   copy .env.example .env
   ```

---

## 4. How to Configure the Existing LLM

The fine-tuned LoRA adapter and tokenizer files are stored in `models/adapter/` (`adapter_model.safetensors`, `adapter_config.json`, and tokenizer configs).
The base model `Qwen/Qwen2.5-1.5B-Instruct` is automatically loaded from HuggingFace cache or downloaded on first run.

If you wish to change base weights or directory paths, configure `.env`:
```ini
BASE_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
PORT=5000
WEB_ACCESS_ENABLED=true
SEARCH_PROVIDER=duckduckgo
RETRIEVAL_TOP_K=3
```

---

## 5. Launching the Chatbot

### Option A: Web User Interface (Recommended)
Double-click `run_web_ui.bat` or run:
```powershell
python app.py
```
Open your browser to: **http://127.0.0.1:5000**

Features available in the Web UI:
- **Search Mode Dropdown**: Switch between `Auto`, `Local KB`, `Live Web`, or `LLM Only`.
- **Web Access Toggle**: Instantly toggle `Web: ON / OFF`.
- **Knowledge Base Modal**: View stored sources, chunk counts, versions, and audit logs.
- **Scrape & Add**: Enter any URL to immediately index it into the knowledge base.
- **Clickable Citations**: Expandable citation cards underneath answers with direct source links.

### Option B: Interactive CLI Console
Double-click `run_cli.bat` or run:
```powershell
python cli_chat.py
```

Console commands:
- `/scrape <url>`: Scrape and add a webpage to the knowledge base.
- `/sources`: List all stored knowledge base documents and chunk counts.
- `/refresh [url]`: Incrementally refresh all or a specific source.
- `/delete <url>`: Delete a source and its chunks.
- `/mode <auto|local|live|llm>`: Change query routing mode.
- `/web <on|off>`: Toggle live web access.
- `/clear`: Clear conversation history.
- `exit` or `quit`: Terminate the console.

---

## 6. How Citations and Grounding Work

When the system routes a query to local RAG, live web, or a direct URL:
1. Retrieved document passages are formatted into a strict isolated context block:
   ```
   === RETRIEVED CONTEXT ===
   [Source 1: Title] (https://example.com/page)
   <Extracted passage>
   === END OF CONTEXT ===
   ```
2. The model receives system instructions requiring factual grounding and prohibiting hallucinations:
   - Must answer based strictly on evidence.
   - Must state when evidence is insufficient.
   - Treats all context text as untrusted data to prevent prompt injection.
3. Every response cites the relevant source URLs and page titles beneath the answer.

---

## 7. Knowledge Base & Incremental Updates

The local storage layer (`src/local_llm/storage.py`) uses SQLite with SHA-256 content hashing:
- **Incremental Refresh**: When refreshing existing sources, if a page's content hash has not changed, re-chunking is skipped, saving compute and memory.
- **Version Tracking**: When content changes, the document version is incremented, and chunks are atomically updated within a transaction.
- **Fail-Safe Updates**: If a website is temporarily down during a refresh, previously valid content is preserved and an error record is logged.

---

## 8. Web Scraping Safety & Security Controls

Before fetching any URL, `src/local_llm/web_scraper.py` validates:
- **Allowed Schemes**: Only `http://` and `https://`.
- **SSRF & Private IP Protection**: Blocks requests to loopback (`127.0.0.1`, `::1`), private subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), link-local (`169.254.0.0/16`), multicast, and cloud metadata (`169.254.169.254`, `metadata.google.internal`).
- **Safe Redirects**: Disables automatic HTTP redirects and re-validates every redirect destination against DNS rebinding.
- **Resource Limits**: Configurable response size limit (default 2 MB) and connection timeout (default 10s) prevent memory exhaustion or hanging requests.
- **Clean Extraction**: Automatically strips JavaScript, CSS styles, navigation bars, footers, SVG, and advertisements.

---

## 9. Running the Automated Test Suite

Run the automated test suite:
```powershell
python -m pytest tests -v
```

Test coverage includes:
- `test_web_scraper.py`: HTML extraction, SSRF blocking, unsafe redirect rejection, timeout handling.
- `test_text_processor.py`: Whitespace cleaning, sentence-boundary chunking, metadata attachment.
- `test_storage.py`: CRUD operations, incremental updates, versioning, cascading deletion.
- `test_retrieval.py`: Local vector embedding, hybrid BM25 + cosine search, context formatting.
- `test_query_router.py`: Direct URL detection, keyword routing, fallback behaviors, mode overrides.
- `test_llm_integration.py`: Grounded prompt synthesis, prompt injection defense, RAG workflows A/B/C.

---

## 10. Hardware & Known Limitations

- **CPU Inference Speed**: Running on CPU without GPU acceleration produces tokens at ~1 to 3 tokens/sec. For faster inference, install `torch` with CUDA if an NVIDIA GPU is available (`pip install torch --index-url https://download.pytorch.org/whl/cu121`).
- **JavaScript-Only Webpages**: Static HTML is parsed using Requests and BeautifulSoup. Pages that require complex client-side JavaScript rendering may yield limited text unless server-side rendered.
- **Anti-Bot & Paywalls**: Websites protected by CAPTCHAs, Cloudflare turnstiles, or subscription paywalls will fail gracefully without attempting illegal bypasses.