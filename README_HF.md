---
title: AlphaSignal
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app_deploy.py
pinned: false
---

# AlphaSignal 📈
### Multi-Agent Financial Intelligence System

Autonomously analyzes SEC 10-K filings using a 5-agent LangGraph pipeline.

**Pipeline:** SEC EDGAR → HTML Parser → RAG (ChromaDB) → Financial Extractor → Sentiment Analyzer → Research Report

**Stack:** LangGraph · ChromaDB · Groq (Llama 3) · FastAPI · Streamlit · ReportLab

Built by Jennica Wang | [GitHub](https://github.com/JennicaANK/alphasignal)