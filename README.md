# 🚀 Local LLM
### A Local AI Chatbot with Custom Web Scraping & Retrieval-Augmented Generation

**A locally running AI chatbot powered by Qwen2.5-1.5B-Instruct, custom PEFT LoRA fine-tuning, hybrid RAG retrieval, and secure web scraping.**

## 🧠 Overview

Local LLM is a Python-based AI chatbot that combines local language-model inference with a custom Retrieval-Augmented Generation (RAG) system.

It can:
- Generate responses using a locally running Qwen language model.
- Search a persistent local knowledge base.
- Scrape and index website content.
- Retrieve relevant information using hybrid BM25 and cosine similarity.
- Generate answers grounded in retrieved evidence.
- Provide clickable source citations.
- Support both Web UI and CLI interaction.

The system allows users to add new knowledge without retraining the language model.

## ✨ Features

- **Local AI Inference:** Qwen2.5-1.5B-Instruct with a custom PEFT LoRA adapter.
- **Retrieval-Augmented Generation:** Retrieve relevant information from locally stored documents.
- **Hybrid Search:** Combine BM25 lexical retrieval with dense cosine similarity.
- **Custom Web Scraper:** Extract and process website content.
- **Live Web Search:** Discover information using DuckDuckGo and Brave Search API.
- **SQLite Knowledge Base:** Persistently store documents, chunks, versions, and crawl history.
- **Intelligent Query Routing:** Choose between LLM-only, Local RAG, Live Web, and Direct URL modes.
- **Source Citations:** Display clickable references for retrieved information.
- **Web Interface:** Interactive browser-based chat with streaming responses.
- **CLI Interface:** Manage sources and interact through the terminal.
- **Security:** SSRF protection, redirect validation, request timeouts, and response-size limits.

## 🏗️ Architecture

```mermaid
flowchart TD
    A[User] --> B[Web UI / CLI]
    B --> C[Query Router]

    C -->|LLM Only| H[Prompt Synthesizer]
    C -->|Local RAG| D[Hybrid Retriever]
    C -->|Live Web / Direct URL| E[Web Discovery & Scraper]

    E --> F[Text Processor]
    F --> G[(SQLite Knowledge Base)]
    G --> D

    D --> H
    H --> I[Qwen2.5 + LoRA]
    I --> J[Answer + Citations]
    J --> B
