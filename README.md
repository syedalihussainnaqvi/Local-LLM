from pathlib import Path

readme = r'''---
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

# Local AI Chatbot with Custom Web Scraping & RAG Integration

<p align="center">
  <strong>A local instruction-tuned AI chatbot with a custom web-scraping pipeline, persistent knowledge base, hybrid retrieval, and evidence-grounded answers.</strong>
</p>

<p align="center">
  <em>Powered by Qwen2.5-1.5B-Instruct, PEFT LoRA, Python, SQLite, and Retrieval-Augmented Generation.</em>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [How the System Works](#how-the-system-works)
- [Core Components](#core-components)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model and LoRA Adapter](#model-and-lora-adapter)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Using the Web Interface](#using-the-web-interface)
- [Using the CLI](#using-the-cli)
- [RAG, Retrieval, and Citations](#rag-retrieval-and-citations)
- [Knowledge Base and Incremental Updates](#knowledge-base-and-incremental-updates)
- [Web Scraping Security](#web-scraping-security)
- [Testing](#testing)
- [Hardware Requirements and Performance](#hardware-requirements-and-performance)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

**Local LLM** is a locally running instruction-tuned chatbot built around **Qwen2.5-1.5B-Instruct** and a fine-tuned **PEFT LoRA adapter**. It combines local language-model inference with a custom web-scraping and Retrieval-Augmented Generation (RAG) pipeline.

The application can discover and fetch website content, extract useful text, divide it into searchable chunks, and store those chunks in a persistent SQLite knowledge base. When a user asks a question, the system can retrieve relevant passages and provide them to the local language model as supporting context.

The model then generates an answer grounded in the retrieved evidence. Where applicable, the application presents clickable source citations so users can inspect the original pages.

A key design goal is to make new information available to the chatbot **without retraining the language model**. Website content is added to the external knowledge base and retrieved at question time.

### What this project is designed to do

- Run an instruction-tuned language model locally.
- Use a custom LoRA adapter with the Qwen2.5-1.5B-Instruct base model.
- Scrape and extract text from supported websites.
- Store website content and metadata in SQLite.
- Retrieve relevant passages using lexical and dense similarity methods.
- Route questions to the appropriate processing path.
- Generate evidence-grounded answers with source references.
- Provide both a browser-based interface and a command-line interface.

---

## Key Features

| Feature | Description |
|---|---|
| Local LLM inference | Runs Qwen2.5-1.5B-Instruct with a PEFT LoRA adapter. |
| Custom web scraping | Fetches web pages and extracts useful semantic content. |
| Retrieval-Augmented Generation | Supplies relevant retrieved passages to the language model at answer time. |
| Persistent knowledge base | Stores documents, chunks, metadata, and crawl history in SQLite. |
| Hybrid retrieval | Combines BM25 lexical retrieval with dense cosine similarity. |
| Intelligent query routing | Routes requests to LLM-only, local RAG, live-web, or direct-URL processing. |
| Web discovery | Supports DuckDuckGo, Brave Search API, and mock offline testing. |
| Source citations | Provides source titles and clickable URLs for retrieved evidence. |
| Incremental refresh | Uses SHA-256 content hashes to avoid unnecessary reprocessing. |
| Grounding instructions | Instructs the model to use retrieved evidence and acknowledge insufficient evidence. |
| Prompt-injection awareness | Treats retrieved website content as untrusted input. |
| Web UI | Provides chat, streaming output, search modes, and knowledge-base management. |
| CLI | Provides terminal-based chat and source-management commands. |
| Scraping safeguards | Includes URL validation, SSRF protections, redirect checks, and resource limits. |

---

## System Architecture

The application is organized as a pipeline connecting the user interface, query router, web discovery and scraping, text processing, SQLite storage, retrieval, prompt synthesis, and local model inference.

```text
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
                 | (Direct URL / Live Web)    | (Local KB Match)           |
                 v                            v                            v
      +----------+----------+      +----------+----------+          [LLM Only]
      |   Web Discovery &   |      |   Hybrid Retriever  |
      |    Safe Scraper     |      | (BM25 + Dense Cosine)|
      +----------+----------+      +----------+----------+
                 |                            |
                 v                            |
      +----------+----------+                 |
      |   Text Processor    |                 |
      | (Sentence Chunking) |                 |
      +----------+----------+                 |
                 |                            |
                 v                            |
      +---------------------+                 |
      | SQLite Local Storage|<----------------+
      |  (rag_storage.db)   |
      +---------------------+
                 |
                 v
      +---------------------+
      |  Prompt Synthesizer |
      |  & Grounding Filter |
      +----------+----------+
                 |
                 v
      +---------------------+
      |    Existing Local   |
      |  Qwen2.5-1.5B + LoRA|
      +----------+----------+
                 |
                 v
      +---------------------+
      | Streaming Answer &  |
      |  Source Citations   |
      +---------------------+
```

### Main processing paths

1. **Direct URL:** A user supplies a URL. The application validates and fetches the page, extracts content, and processes it for the request.
2. **Live Web:** The discovery component searches for relevant pages, after which the scraper retrieves and processes supported results.
3. **Local RAG:** The retriever searches the existing SQLite knowledge base for relevant passages.
4. **LLM Only:** The request is sent to the local model without retrieving external or local knowledge-base content.

The query router selects a path based on the request and the selected application mode.

---

## How the System Works

The typical RAG workflow is:

1. **Receive a question** — The user submits a prompt through the Web UI or CLI.
2. **Analyze intent** — The query router identifies the requested processing mode.
3. **Discover or select sources** — Depending on the route, the application uses a supplied URL, web discovery, or the existing local knowledge base.
4. **Fetch and extract content** — The scraper retrieves supported web pages and extracts useful text.
5. **Normalize and chunk text** — The text processor cleans the extracted content and divides it into sentence-aware chunks.
6. **Store knowledge** — Documents, chunks, metadata, and content hashes are stored in SQLite.
7. **Retrieve relevant evidence** — The hybrid retriever searches for passages relevant to the user's question.
8. **Build a grounded prompt** — Retrieved passages are placed in a clearly delimited context block with source metadata.
9. **Generate an answer** — The local Qwen2.5 model with the LoRA adapter generates the response.
10. **Present the result** — The interface streams the answer and displays relevant source citations where available.

**Important:** RAG adds external knowledge at inference time. Adding a website to the knowledge base does not, by itself, fine-tune or retrain the model.

---

## Core Components

The main application package is located in `src/local_llm/`.

| Component | Responsibility |
|---|---|
| `config.py` | Centralized configuration management, safe defaults, and environment-variable overrides. |
| `web_scraper.py` | SSRF-protected web fetching, redirect re-validation, response-size limits, robots.txt support, and semantic HTML extraction. |
| `text_processor.py` | Whitespace normalization, sentence-boundary-aware chunking, and source metadata tracking. |
| `storage.py` | Transactional SQLite persistence, incremental updates, document versioning, cascading deletion, and crawl history. |
| `embeddings.py` | CPU-efficient local hashing embedder that creates normalized dense vectors without requiring external embedding-model downloads. |
| `retrieval.py` | Hybrid BM25 lexical and dense cosine retrieval, threshold filtering, and context formatting. |
| `query_router.py` | Intent analysis and routing between `llm_only`, `local_rag`, `live_web`, and `direct_url`. |
| `web_discovery.py` | Search and discovery integrations for DuckDuckGo, Brave Search API, and mock offline testing. |
| `rag_pipeline.py` | Orchestrates routing, scraping, storage, retrieval, and grounding instructions. |
| `llm_engine.py` | Persistent, thread-safe inference manager for the Qwen2.5-1.5B base model and LoRA adapter. |
| `app.py` | Flask web-application backend. |
| `templates/index.html` | Browser interface, including themes, streaming chat, knowledge-base modal, and citation cards. |
| `cli_chat.py` | Interactive command-line chat and knowledge-base management. |

### Component responsibilities in more detail

#### Configuration — `config.py`

Centralizes application settings and provides safe defaults. Environment variables can override supported settings, allowing local configuration without changing application source code.

#### Web scraper — `web_scraper.py`

Responsible for safely fetching supported web pages. It validates URLs and redirect destinations, applies request limits, supports robots.txt handling, and extracts semantic page content.

#### Text processor — `text_processor.py`

Normalizes extracted text and splits it into sentence-aware chunks. Chunk metadata can include the source URL, page title, section information, and content hash.

#### Storage — `storage.py`

Maintains the persistent SQLite knowledge base. It supports transactional updates, incremental refreshes, version tracking, cascading deletion, and crawl-history records.

#### Embeddings — `embeddings.py`

Creates normalized dense vectors locally using a hashing-based approach. The documented implementation does not require downloading a separate embedding model.

#### Retrieval — `retrieval.py`

Finds useful passages by combining lexical BM25 retrieval with dense cosine similarity. It can filter results using configured thresholds and format retrieved passages for the language-model prompt.

#### Query router — `query_router.py`

Analyzes the user's request and chooses among the supported routes:

- `llm_only`
- `local_rag`
- `live_web`
- `direct_url`

#### RAG pipeline — `rag_pipeline.py`

Coordinates the components required to process a query, including routing, web retrieval, content storage, local retrieval, and grounding instructions.

#### LLM engine — `llm_engine.py`

Manages local inference using the Qwen2.5-1.5B-Instruct base model and the project's PEFT LoRA adapter. The documented engine is designed to be persistent and thread-safe.

---

## Project Structure

The documented project layout is:

```text
.
├── src/
│   └── local_llm/
│       ├── agent/
│       │   └── ...
│       ├── templates/
│       │   └── index.html
│       ├── config.py
│       ├── web_scraper.py
│       ├── text_processor.py
│       ├── storage.py
│       ├── embeddings.py
│       ├── retrieval.py
│       ├── query_router.py
│       ├── web_discovery.py
│       ├── rag_pipeline.py
│       ├── llm_engine.py
│       ├── app.py
│       └── cli_chat.py
├── models/
│   └── adapter/
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       └── tokenizer files
├── tests/
├── eval/
├── docs/
├── scripts/
│   └── model/
├── dev/
│   └── scratch/
├── data/
│   └── rag_storage.db
├── app.py
├── cli_chat.py
├── agent_cli.py
├── requirements.txt
├── .env.example
├── run_web_ui.bat
└── run_cli.bat
```

The root-level `app.py`, `cli_chat.py`, and `agent_cli.py` are compatibility launchers. New application code belongs in `src/local_llm/`.

> The tree above documents the intended layout. Runtime-generated files, local environment files, and optional development files may not exist in a fresh clone.

---

## Technology Stack

| Technology | Role in the project |
|---|---|
| Python | Main application language |
| Qwen2.5-1.5B-Instruct | Base instruction-tuned language model |
| PEFT | Parameter-efficient model adaptation |
| LoRA | Fine-tuned adapter approach |
| PyTorch | Model inference |
| Hugging Face Transformers | Model loading and generation |
| SQLite | Persistent local knowledge base |
| BM25 | Lexical information retrieval |
| Cosine similarity | Dense-vector similarity |
| Flask | Web application backend |
| HTML, CSS, JavaScript | Web interface |
| Requests | HTTP requests |
| BeautifulSoup | HTML parsing and extraction |
| DuckDuckGo | Web discovery provider |
| Brave Search API | Optional web discovery provider |

---

## Prerequisites

### Supported operating systems

- Windows
- Linux
- macOS

### Software

- Python **3.10–3.14**, as specified by the project documentation.
- Git.
- `pip`.
- A terminal or command prompt.

### Compute

- **CPU:** Supported, with a PyTorch CPU fallback.
- **GPU:** CUDA acceleration is used when a compatible CUDA environment is available.
- **RAM:** 8 GB is the documented recommended minimum.
- **Model footprint:** The documented base-model footprint is approximately 3.1 GB. Additional memory and disk space are required for dependencies, runtime, and application data.

The exact performance depends on the hardware, PyTorch build, model-loading configuration, and generation settings.

---

## Installation

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username and adjust the repository name if yours is different.

```bash
git clone https://github.com/YOUR-USERNAME/Local-LLM.git
cd Local-LLM
```

If the repository is already on your computer, open a terminal in the project directory instead.

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```bat
venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Create the environment file

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
```

**Windows Command Prompt:**

```bat
copy .env.example .env
```

**Linux / macOS:**

```bash
cp .env.example .env
```

### 6. Review local configuration

Open `.env` and update the settings required for your environment. See [Configuration](#configuration).

---

## Model and LoRA Adapter

The chatbot uses the following base model:

**`Qwen/Qwen2.5-1.5B-Instruct`**

The project also includes a fine-tuned **PEFT LoRA adapter**.

The documented adapter directory is:

```text
models/adapter/
├── adapter_model.safetensors
├── adapter_config.json
└── tokenizer files
```

The base model can be loaded from the Hugging Face cache or downloaded when the application first needs it, depending on the local environment and model-loading configuration.

### Base model configuration

```env
BASE_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
```

### Important distinction: base model, adapter, and RAG

- **Base model:** Qwen2.5-1.5B-Instruct provides the underlying language-model capabilities.
- **LoRA adapter:** Supplies the project's parameter-efficient fine-tuned model adaptation.
- **RAG knowledge base:** Supplies external or user-added information at inference time through retrieval.

The RAG knowledge base is separate from the model weights and adapter. Adding or refreshing website content updates the knowledge base; it does not automatically modify the model.

---

## Configuration

The application supports centralized configuration with environment-variable overrides.

Example `.env`:

```ini
BASE_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
PORT=5000
WEB_ACCESS_ENABLED=true
SEARCH_PROVIDER=duckduckgo
RETRIEVAL_TOP_K=3
```

| Variable | Example | Purpose |
|---|---|---|
| `BASE_MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | Selects the base language model. |
| `PORT` | `5000` | Port used by the web application. |
| `WEB_ACCESS_ENABLED` | `true` | Enables or disables web access. |
| `SEARCH_PROVIDER` | `duckduckgo` | Selects the configured search provider. |
| `RETRIEVAL_TOP_K` | `3` | Sets the number of top retrieval results requested by the documented configuration. |

Only add environment variables supported by the current project configuration. The `.env.example` file is the reference for the settings available in your checkout.

**Security note:** Keep private API keys and local secrets in `.env`. Do not commit `.env` to a public repository. Commit `.env.example` with placeholder values instead.

---

## Running the Application

### Option A: Web interface

The browser-based interface is the recommended way to interact with the chatbot.

Run:

```bash
python app.py
```

Or, on Windows, use the provided launcher:

```bat
run_web_ui.bat
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

### Option B: CLI interface

Run:

```bash
python cli_chat.py
```

Or use the Windows launcher:

```bat
run_cli.bat
```

The CLI provides an interactive terminal-based chatbot and commands for managing knowledge-base sources.

---

## Using the Web Interface

The documented Web UI includes the following capabilities:

### Search mode selector

Select how the application should process a question:

| Mode | Behavior |
|---|---|
| `Auto` | Lets the query router select a processing route. |
| `Local KB` | Uses the local knowledge base for retrieval. |
| `Live Web` | Uses the web discovery and retrieval workflow. |
| `LLM Only` | Uses local language-model inference without RAG retrieval. |

### Web access toggle

The interface provides a Web Access toggle to turn web access on or off.

### Knowledge Base modal

The Knowledge Base interface provides visibility into stored sources, including documented information such as:

- Stored source pages.
- Chunk counts.
- Document versions.
- Crawl or audit history.

### Scrape and Add

Use the URL input to submit a website for scraping and knowledge-base indexing.

### Source citations

When source evidence is available, the interface displays citation cards containing source information and direct links to the original pages.

---

## Using the CLI

Start the command-line interface:

```bash
python cli_chat.py
```

### Available commands

| Command | Description |
|---|---|
| `/scrape <url>` | Scrape a webpage and add it to the knowledge base. |
| `/sources` | List stored knowledge-base documents and chunk counts. |
| `/refresh` | Refresh all supported stored sources. |
| `/refresh <url>` | Refresh a specific source. |
| `/delete <url>` | Delete a source and its associated chunks. |
| `/mode <auto\|local\|live\|llm>` | Change the query-routing mode. |
| `/web <on\|off>` | Enable or disable web access. |
| `/clear` | Clear conversation history. |
| `exit` or `quit` | Exit the CLI. |

### Example CLI session

```text
> /scrape https://example.com
> /sources
> /mode local
> What information is available about this topic?
> /refresh https://example.com
> /delete https://example.com
> exit
```

The example demonstrates the documented command format. Replace the example URL with a website you are permitted to access.

---

## RAG, Retrieval, and Citations

Retrieval-Augmented Generation allows the application to provide relevant external information to the local language model without changing the model's weights.

### 1. Content retrieval

Depending on the selected route, the system can retrieve content from:

- The existing local SQLite knowledge base.
- Pages discovered through the configured web search provider.
- A URL supplied directly by the user.

### 2. Context construction

Retrieved passages are formatted into an isolated context block with source information.

Example format:

```text
=== RETRIEVED CONTEXT ===

[Source 1: Example Page]
(https://example.com/page)

Extracted passage from the source page.

[Source 2: Another Example Page]
(https://example.org/article)

Another relevant extracted passage.

=== END OF CONTEXT ===
```

### 3. Grounded generation

The model receives instructions to:

- Base factual answers on the retrieved evidence.
- Acknowledge when the available evidence is insufficient.
- Avoid presenting unsupported claims as verified facts.
- Treat retrieved content as untrusted input rather than executable instructions.

### 4. Citation presentation

The application associates retrieved information with its source title and URL. Relevant source citations can then be displayed beneath the generated answer.

**Note:** Citations make it possible to inspect the source material; users should still open and verify important information against the original page.

---

## Knowledge Base and Incremental Updates

The persistent knowledge base is stored in SQLite.

Documented database location:

```text
data/rag_storage.db
```

The storage layer uses SHA-256 content hashing and supports incremental updates.

### Incremental refresh

When a stored source is refreshed:

1. The application fetches and processes the source.
2. A content hash is calculated.
3. If the content hash has not changed, unnecessary re-chunking is skipped.
4. If the content has changed, the document version is incremented.
5. Updated chunks are written transactionally.

### Version tracking

When page content changes, the storage layer tracks the updated document version and replaces its associated chunks within a database transaction.

### Fail-safe refresh behavior

If a website is temporarily unavailable during a refresh, the documented behavior is to preserve the previous valid content and record the error rather than replacing good stored data with an empty result.

### Storage capabilities

- Persistent document storage.
- Chunk-level storage.
- Source metadata.
- SHA-256 content hashes.
- Incremental refresh.
- Document versioning.
- Transactional updates.
- Cascading deletion.
- Crawl and audit history.

---

## Web Scraping Security

The scraper is designed with safeguards against unsafe URL fetching and excessive resource consumption.

### URL scheme validation

Only the following URL schemes are accepted:

```text
http://
https://
```

Other schemes are rejected.

### SSRF and private-network protection

The documented scraper blocks requests to destinations such as:

- Loopback addresses, including `127.0.0.1` and `::1`.
- Private IPv4 ranges:
  - `10.0.0.0/8`
  - `172.16.0.0/12`
  - `192.168.0.0/16`
- Link-local addresses, including `169.254.0.0/16`.
- Multicast destinations.
- Cloud metadata destinations, including `169.254.169.254` and `metadata.google.internal`.

### Redirect validation

Automatic redirects are disabled. Each redirect destination is validated before the application follows it.

The documented implementation also includes protection intended to address DNS-rebinding risks.

### Resource limits

The documented default limits are:

| Control | Default |
|---|---:|
| Maximum response size | 2 MB |
| Connection timeout | 10 seconds |

These limits help reduce the risk of excessive memory use and requests that remain open for too long.

### HTML extraction and cleanup

The scraper removes or excludes page elements that are generally not useful as knowledge-base content, including:

- JavaScript.
- CSS styles.
- Navigation elements.
- Footers.
- SVG elements.
- Advertisements.

It also supports semantic HTML extraction and robots.txt handling.

### Website access limitations

The scraper does not attempt to bypass CAPTCHAs, Cloudflare challenges, subscription paywalls, or other access controls. Such pages may fail gracefully or return limited content.

---

## Testing

Run the documented test suite with:

```bash
python -m pytest tests -v
```

The test coverage described by the project includes:

| Test module | Areas covered |
|---|---|
| `test_web_scraper.py` | HTML extraction, SSRF blocking, unsafe redirect rejection, and timeout handling. |
| `test_text_processor.py` | Whitespace normalization, sentence-boundary chunking, and metadata attachment. |
| `test_storage.py` | CRUD operations, incremental updates, versioning, and cascading deletion. |
| `test_retrieval.py` | Local embeddings, hybrid BM25 and cosine retrieval, and context formatting. |
| `test_query_router.py` | Direct URL detection, keyword routing, fallback behavior, and mode overrides. |
| `test_llm_integration.py` | Grounded prompt synthesis, prompt-injection defenses, and RAG workflows. |

Run tests from the project root so that the test runner can resolve the project package and test paths.

---

## Hardware Requirements and Performance

The application supports CPU inference and can use CUDA when a compatible NVIDIA GPU environment is available.

| Resource | Project documentation |
|---|---|
| Operating system | Windows, Linux, macOS |
| Python | 3.10–3.14 |
| RAM | 8 GB recommended minimum |
| Base model footprint | Approximately 3.1 GB |
| CPU inference | Supported |
| GPU inference | CUDA when available |

### CPU performance

The project documentation estimates CPU generation performance at approximately **1–3 tokens per second** for the documented configuration. Actual throughput depends on the processor, memory, PyTorch build, and generation settings.

### GPU acceleration

If a compatible NVIDIA GPU and CUDA environment are available, install a PyTorch build that matches the system's CUDA requirements. Follow the official PyTorch installation selector for the correct command:

https://pytorch.org/get-started/locally/

Do not install a CUDA-specific build unless it is compatible with the installed NVIDIA driver and hardware.

---

## Known Limitations

### JavaScript-heavy websites

The documented scraper uses Requests and BeautifulSoup to process static HTML. Websites that render their main content only through client-side JavaScript may provide limited extracted text.

### Anti-bot systems and paywalls

CAPTCHAs, Cloudflare challenges, and subscription paywalls can prevent content retrieval. The scraper does not attempt to bypass these controls.

### CPU inference speed

Inference on CPU can be slow compared with GPU execution. The documented estimate is approximately 1–3 tokens per second for the stated configuration.

### Model limitations

A small language model can still generate inaccurate or incomplete responses. Retrieval and citations help provide supporting evidence, but they do not guarantee that every generated statement is correct.

### Source availability

Live-web retrieval depends on website availability, network access, and the configured search provider. Previously stored valid content is intended to remain available if a source temporarily becomes unavailable during refresh.

---

## Troubleshooting

### The model takes a long time to load

- Confirm that the base model can be accessed or is already available in the Hugging Face cache.
- Check that sufficient RAM and disk space are available.
- Allow the first model download to complete before retrying.

### The application cannot find the adapter

- Confirm that the adapter files exist in `models/adapter/`.
- Check that `adapter_model.safetensors` and `adapter_config.json` are present.
- Confirm that the configured adapter path matches the actual project layout.

### The Web UI does not open

- Confirm that the application started without an error.
- Check the configured `PORT` value.
- Open the matching local address, such as `http://127.0.0.1:5000`.

### A webpage cannot be scraped

- Confirm that the URL uses HTTP or HTTPS.
- Check whether the website is publicly accessible.
- Check whether the site uses JavaScript-only rendering, anti-bot protection, or a paywall.
- Remember that private, loopback, link-local, and cloud metadata destinations are intentionally blocked.

### No useful local RAG results are returned

- Add the relevant source to the knowledge base.
- Check the source list using `/sources` in the CLI.
- Confirm that the source was fetched and indexed successfully.
- Try a more specific query.
- Review the configured retrieval settings.

---

## Roadmap

Potential future improvements for the project include:

- [ ] Support for additional local language models.
- [ ] More advanced document-ingestion formats.
- [ ] Retrieval-quality evaluation and tuning.
- [ ] Improved handling of JavaScript-rendered websites.
- [ ] Knowledge-base import and export.
- [ ] Expanded evaluation datasets.
- [ ] Additional automated integration tests.
- [ ] Deployment and packaging improvements.
- [ ] Performance optimization for local inference.

Items in this roadmap are ideas for future development and should not be interpreted as already implemented features.

---

## Contributing

Contributions, bug reports, and suggestions are welcome.

A typical contribution workflow:

1. Fork the repository.
2. Create a feature branch.
3. Make a focused change.
4. Run the relevant tests.
5. Commit your changes with a clear message.
6. Open a Pull Request describing the change.

For larger changes, consider opening an issue first to discuss the proposed implementation.

---

## License

No license is specified in the supplied project documentation.

Before making the repository public for reuse, add a `LICENSE` file and update this section with the selected license. Without an explicit license, others generally do not receive permission to reuse, modify, or distribute the project's code.

---

## Author

**Syed Ali Hussain Naqvi**

Artificial Intelligence Student | AI Developer | Python Developer

- GitHub: `https://github.com/YOUR-USERNAME`
- LinkedIn: https://www.linkedin.com/in/ali-hussain-gdesigner/

---

<p align="center">
  <strong>Local Intelligence. Searchable Knowledge. Evidence-Grounded Answers.</strong>
  <br />
  Built with Python, Qwen2.5, PEFT, SQLite, and RAG.
</p>
'''

path = Path("/mnt/data/README_Local_LLM_Detailed.md")
path.write_text(readme, encoding="utf-8")
print(f"Created: {path}")
print(f"Size: {path.stat().st_size:,} bytes")
