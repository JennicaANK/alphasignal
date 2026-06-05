# AlphaSignal 📈
### Multi-Agent Financial Intelligence System

## Overview
AlphaSignal is a production-grade multi-agent AI system that autonomously 
analyzes SEC financial filings (10-K, 10-Q) to extract insights, detect 
sentiment trends, and generate research reports — with a confidence score 
on every single claim.

## The Problem
Financial analysts spend 60% of their time manually reading 200+ page SEC 
filings. Existing AI tools hallucinate financial figures with no way to 
know when to trust them.

## The Solution
A self-correcting multi-agent pipeline where five specialized AI agents 
collaborate: one fetches documents, one extracts financials, one analyzes 
sentiment, one self-checks low-confidence claims, and one writes the final 
report. Every output includes an uncertainty score.

## Architecture
User Query → Document Fetcher Agent
→ Financial Extractor Agent (RAG)
→ Sentiment Analyzer Agent
→ Self-Checker Agent (re-retrieves if confidence < 0.7)
→ Report Writer Agent
→ PDF Report + Streamlit Dashboard


## Tech Stack
| Layer | Tool |
|---|---|
| Agent Orchestration | LangGraph |
| LLM | Groq (Llama 3 — free) |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers |
| Data Source | SEC EDGAR API (free) |
| Backend | FastAPI |
| Frontend | Streamlit + Plotly |
| Deployment | Docker + Hugging Face Spaces |

## Key Features
- Multi-agent architecture with 5 specialized agents
- Self-correcting RAG with confidence scoring
- Uncertainty quantification on every extracted claim
- Automated PDF research report generation
- Live financial trend visualization

## Project Structure
alphasignal/
├── data/
│   ├── raw/          # SEC filings (gitignored)
│   └── processed/    # Cleaned text chunks
├── notebooks/        # EDA and exploration
├── src/
│   ├── agents/       # LangGraph agent nodes
│   ├── rag/          # RAG pipeline and retrieval
│   └── utils/        # Shared helper functions
├── outputs/          # Charts and exports
├── reports/          # Generated PDF reports
├── app.py            # Streamlit dashboard
├── main.py           # FastAPI entry point
├── requirements.txt
└── README.md


## Status
🚧 Active development — building daily.


