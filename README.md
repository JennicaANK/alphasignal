---
title: AlphaSignal
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_file: app_deploy.py
pinned: false
---

# AlphaSignal 📈
### Production-Grade Multi-Agent Financial Intelligence System

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-HuggingFace-yellow)](https://jennicawang-alphasignal.hf.space)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-purple)](https://langchain-ai.github.io/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**[🚀 Try it live →](https://jennicawang-alphasignal.hf.space)**

AlphaSignal autonomously analyzes SEC 10-K financial filings using a
self-correcting multi-agent pipeline. Input a ticker symbol — get a
full research report with structured financial metrics, management
sentiment analysis, and confidence scores on every claim.

---

## The Problem

Financial analysts spend 60% of their time manually reading 200+ page
SEC filings. Existing AI tools hallucinate financial figures with no
way to know when to trust them.

## The Solution

A production-grade multi-agent system where five specialized AI agents
collaborate — each with a single responsibility — to produce verified,
cited, confidence-scored financial intelligence.

---

## Architecture

User Input (ticker symbol)

│

▼

┌─────────────────────────────────────────────────────────────┐

│                    LangGraph Pipeline                        │

│                                                             │

│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │

│  │   Agent 1    │───▶│   Agent 2    │───▶│   Agent 3    │  │

│  │  Document    │    │  Financial   │    │  Sentiment   │  │

│  │  Fetcher     │    │  Extractor   │    │  Analyzer    │  │

│  └──────────────┘    └──────────────┘    └──────────────┘  │

│         │                   │                   │           │

│    SEC EDGAR           RAG Pipeline       Loughran-         │

│    10-K filing         ChromaDB +         McDonald +        │

│    HTML parser         BM25 reranker      Groq LLM          │

│                                                             │

│                                   ┌──────────────┐          │

│                              ┌───▶│   Agent 4    │          │

│                              │    │ Self-Checker │          │

│                              │    │ Confidence   │          │

│                              │    │   Scoring    │          │

│                              │    └──────┬───────┘          │

│                              │           │                   │

│                   (re-extract if     ┌───▼──────────┐       │

│                   confidence < 0.5)  │   Agent 5    │       │

│                              └───── │    Report    │       │

│                                     │    Writer    │       │

│                                     └──────────────┘       │

└─────────────────────────────────────────────────────────────┘

│

▼

Research Report + PDF + Dashboard

---

## Key Features

- **Multi-agent orchestration** — 5 specialized LangGraph agents with
  shared state and conditional self-correcting loop
- **Hybrid RAG** — BM25 + semantic retrieval with re-ranking,
  self-correcting when confidence < 0.5
- **Uncertainty quantification** — every claim has a confidence score
  combining retrieval quality and answer grounding
- **Financial extraction** — structured JSON output with 3-year
  comparisons and YoY change calculations
- **Sentiment analysis** — Loughran-McDonald financial lexicon +
  LLM dimensional analysis on MD&A section
- **Production deployment** — FastAPI backend, Streamlit dashboard,
  Docker containerization, live on Hugging Face Spaces
- **38 unit tests** — 100% passing, covering all core modules

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM Inference | Groq (Llama-3.1-8b-instant) — free tier |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Retrieval | Hybrid BM25 + semantic with re-ranking |
| Data Source | SEC EDGAR API (free, no key required) |
| Backend | FastAPI with async background tasks |
| Frontend | Streamlit + Plotly |
| PDF Export | ReportLab |
| Containerization | Docker + docker-compose |
| Deployment | Hugging Face Spaces |
| Testing | pytest (38 tests) |

---

## Project Structure

alphasignal/

├── src/

│   ├── agents/

│   │   ├── graph.py        # LangGraph pipeline

│   │   ├── nodes.py        # 5 agent implementations

│   │   └── state.py        # Shared state schema

│   ├── rag/

│   │   ├── chunker.py      # Sliding window chunker

│   │   ├── reranker.py     # Hybrid BM25 + semantic

│   │   └── vector_store.py # ChromaDB operations

│   └── utils/

│       ├── sec_fetcher.py  # SEC EDGAR API client

│       ├── text_parser.py  # HTML parser + cleaner

│       ├── financial_extractor.py # Structured extraction

│       ├── sentiment_analyzer.py  # Loughran-McDonald + LLM

│       └── pdf_generator.py       # ReportLab PDF builder

├── tests/                  # 38 unit tests

├── main.py                 # FastAPI backend

├── app.py                  # Streamlit dashboard (local)

├── app_deploy.py           # Streamlit (HF Spaces)

├── Dockerfile

└── docker-compose.yml

---

## Running Locally

**1. Clone and set up:**
```bash
git clone https://github.com/JennicaANK/alphasignal.git
cd alphasignal
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Add environment variables:**
```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_key_here" > .env
echo "SEC_USER_AGENT=AlphaSignal your@email.com" >> .env
```

**3. Start the API:**
```bash
uvicorn main:app --reload --port 8000
```

**4. Start the dashboard:**
```bash
streamlit run app.py
```

**5. Or use Docker:**
```bash
docker-compose up --build
```

---

## Run the Pipeline

```bash
# Analyze any public company
python -m src.agents.graph
```

Or via the API:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

---

## Run Tests

```bash
python -m pytest tests/ -v --cov=src
# 38 passed in ~1.2s
```

---

## Pipeline Output Example

Running AlphaSignal on Apple Inc. (AAPL) FY2025 10-K:

| Metric | Value |
|---|---|
| Total Revenue | $416,161M (+6.4% YoY) |
| Net Income | $112,010M (+19.5% YoY) |
| Services Revenue | $109,158M (+13.5% YoY) |
| Management Tone | NEUTRAL (5/10) |
| Pipeline Confidence | 1.0000 [HIGH] |

---

## Built By

**Aye Nyein Kyaw (Jennica)**
B.S. Data Science, San Jose State University (May 2026)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-ayenyeinkyaw-blue)](https://linkedin.com/in/ayenyeinkyaw)
[![GitHub](https://img.shields.io/badge/GitHub-JennicaANK-black)](https://github.com/JennicaANK)

---

*Data source: SEC EDGAR (free public API)*
*LLM: Groq free tier (Llama-3.1-8b-instant)*
*Total project cost: $0*
