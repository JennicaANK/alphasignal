"""
AlphaSignal Agent Nodes
-------------------------
Each function is one node in the LangGraph pipeline.
Nodes receive the full state, do their work, and return
a dict of fields to update in the state.

Progress:
    Day 11  fetch_documents     ✅ implemented
    Day 12  extract_financials  ✅ implemented
    Day 13  analyze_sentiment   🔧 stub
    Day 14  check_confidence    🔧 stub
    Day 15  write_report        🔧 stub
"""

import os
from src.agents.state import AlphaSignalState

# ── Module imports ─────────────────────────────────────────────────────────────
# Import here so all agents share the same loaded modules.
# Lazy imports inside each function would reload modules on every call.
from src.utils.sec_fetcher   import download_10k_text, save_metadata
from src.utils.text_parser   import parse_filing
from src.rag.chunker         import load_sections, chunk_sections, save_chunks
from src.rag.vector_store    import add_chunks_to_store


# ── Agent 1: Document Fetcher ─────────────────────────────────────────────────
def fetch_documents(state: AlphaSignalState) -> dict:
    """
    Fetches, parses, chunks, and embeds a 10-K filing end to end.

    Connects four modules built in Weeks 1-2:
        sec_fetcher  → downloads real 10-K from SEC EDGAR API
        text_parser  → strips HTML, cleans noise, extracts sections
        chunker      → splits sections into overlapping chunks
        vector_store → embeds chunks using sentence-transformers + ChromaDB

    On success: updates state with all file paths and company metadata.
    On failure: logs the error and passes state through so the pipeline
                can continue (or fail gracefully downstream).
    """
    ticker      = state["ticker"]
    filing_date = state.get("filing_date")

    print(f"\n  [Agent 1] Document Fetcher — {ticker}")

    try:
        # ── Step 1: Download 10-K from SEC EDGAR ──────────────────────────────
        print(f"  [Agent 1] Downloading 10-K from SEC EDGAR...")
        metadata = download_10k_text(ticker)
        save_metadata(metadata)

        actual_filing_date = metadata["filing_date"]
        print(f"  [Agent 1] Filing date: {actual_filing_date}")
        print(f"  [Agent 1] Company:     {metadata['company_name']}")
        print(f"  [Agent 1] File size:   {metadata['file_size_kb']} KB")

        # ── Step 2: Parse HTML → clean text + sections ────────────────────────
        print(f"  [Agent 1] Parsing and cleaning filing...")
        parse_result = parse_filing(metadata["file_path"])

        print(f"  [Agent 1] Sections extracted: {parse_result['sections_found']}")
        print(f"  [Agent 1] Noise removed:       {parse_result['noise_removed']}")

        # ── Step 3: Chunk sections ─────────────────────────────────────────────
        print(f"  [Agent 1] Chunking sections...")
        sections    = load_sections(parse_result["sections_path"])
        chunks      = chunk_sections(sections, ticker, actual_filing_date)
        chunks_path = save_chunks(chunks)

        print(f"  [Agent 1] Chunks created: {len(chunks)}")

        # ── Step 4: Embed and store in ChromaDB ───────────────────────────────
        print(f"  [Agent 1] Embedding and storing in ChromaDB...")
        add_chunks_to_store(chunks, reset=True)
        print(f"  [Agent 1] Vector store ready.")

        return {
            "company_name":      metadata["company_name"],
            "cik":               metadata["cik"],
            "filing_date":       actual_filing_date,
            "raw_filing_path":   metadata["file_path"],
            "clean_text_path":   parse_result["clean_text_path"],
            "sections_path":     parse_result["sections_path"],
            "chunks_path":       chunks_path,
            "completed_steps":   state.get("completed_steps", []) + ["fetch_documents"],
            "current_step":      "extract_financials",
            "errors":            state.get("errors", []),
        }

    except Exception as e:
        error_msg = f"fetch_documents failed: {str(e)}"
        print(f"  [Agent 1] ERROR: {error_msg}")

        # Return error but don't crash the pipeline
        return {
            "errors":          state.get("errors", []) + [error_msg],
            "completed_steps": state.get("completed_steps", []) + ["fetch_documents_error"],
            "current_step":    "extract_financials",
        }


# ── Agent 2: Financial Extractor ──────────────────────────────────────────────
def extract_financials(state: AlphaSignalState) -> dict:
    """
    Extracts structured financial metrics from the 10-K filing.
    Connects financial_extractor.py into the agent pipeline.

    Inputs:  state["ticker"], state["filing_date"]
    Outputs: state["financials"], state["financials_path"]
    """
    ticker      = state.get("ticker")
    filing_date = state.get("filing_date")
    recheck     = state.get("recheck_count", 0)

    print(f"\n  [Agent 2] Financial Extractor — {ticker}")
    if recheck > 0:
        print(f"  [Agent 2] Re-extraction attempt #{recheck}")

    try:
        from src.utils.financial_extractor import extract_financials as run_extraction

        financials = run_extraction(
            ticker      = ticker,
            filing_date = filing_date,
        )

        if not financials:
            raise ValueError("Extraction returned empty result")

        # Pull key metrics for logging
        inc  = financials.get("income_statement", {})
        rev  = inc.get("total_net_sales", {}).get("year_1", "N/A")
        ni   = inc.get("net_income",      {}).get("year_1", "N/A")

        print(f"  [Agent 2] Revenue:    {rev:,.0f}M" if isinstance(rev, float) else f"  [Agent 2] Revenue:    {rev}")
        print(f"  [Agent 2] Net income: {ni:,.0f}M"  if isinstance(ni,  float) else f"  [Agent 2] Net income: {ni}")

        financials_path = f"data/processed/{ticker}_{filing_date}_financials.json"

        return {
            "financials":      financials,
            "financials_path": financials_path,
            "completed_steps": state.get("completed_steps", []) + ["extract_financials"],
            "current_step":    "analyze_sentiment",
            "errors":          state.get("errors", []),
        }

    except Exception as e:
        error_msg = f"extract_financials failed: {str(e)}"
        print(f"  [Agent 2] ERROR: {error_msg}")

        return {
            "financials":      None,
            "errors":          state.get("errors", []) + [error_msg],
            "completed_steps": state.get("completed_steps", []) + ["extract_financials_error"],
            "current_step":    "analyze_sentiment",
        }


# ── Agent 3: Sentiment Analyzer ───────────────────────────────────────────────
def analyze_sentiment(state: AlphaSignalState) -> dict:
    """
    Analyzes tone and language in the MD&A section.
    Will be implemented Day 13.
    """
    print(f"\n  [Agent 3] Sentiment Analyzer — {state['ticker']}")
    print(f"  [Agent 3] STUB — will be implemented Day 13")

    return {
        "completed_steps": state.get("completed_steps", []) + ["analyze_sentiment"],
        "current_step":    "check_confidence",
        "errors":          state.get("errors", []),
    }


# ── Agent 4: Self-Checker ─────────────────────────────────────────────────────
def check_confidence(state: AlphaSignalState) -> dict:
    """
    Evaluates pipeline quality and triggers re-extraction if needed.
    Will be implemented Day 14.
    """
    print(f"\n  [Agent 4] Self-Checker — evaluating pipeline confidence")
    print(f"  [Agent 4] STUB — will be implemented Day 14")

    return {
        "confidence_score": 0.75,
        "confidence_label": "HIGH",
        "needs_recheck":    False,
        "recheck_count":    state.get("recheck_count", 0),
        "completed_steps":  state.get("completed_steps", []) + ["check_confidence"],
        "current_step":     "write_report",
        "errors":           state.get("errors", []),
    }


# ── Agent 5: Report Writer ────────────────────────────────────────────────────
def write_report(state: AlphaSignalState) -> dict:
    """
    Generates the final research report.
    Will be implemented Day 15.
    """
    print(f"\n  [Agent 5] Report Writer — generating report")
    print(f"  [Agent 5] STUB — will be implemented Day 15")

    return {
        "report":          f"[STUB] Report for {state['ticker']} — to be generated Day 15",
        "completed_steps": state.get("completed_steps", []) + ["write_report"],
        "current_step":    "complete",
        "errors":          state.get("errors", []),
    }


# ── Routing Function ──────────────────────────────────────────────────────────
def route_after_confidence_check(state: AlphaSignalState) -> str:
    """
    Routes after self-checker based on confidence score.
    """
    needs_recheck = state.get("needs_recheck", False)
    recheck_count = state.get("recheck_count", 0)
    confidence    = state.get("confidence_score", 1.0)

    if needs_recheck and recheck_count < 2:
        print(f"\n  [Router] Confidence {confidence:.2f} too low — re-extracting (attempt {recheck_count + 1})")
        return "extract_financials"
    elif recheck_count >= 2:
        print(f"\n  [Router] Max rechecks reached — proceeding with warning")
        return "write_report"
    else:
        print(f"\n  [Router] Confidence {confidence:.2f} — proceeding to report")
        return "write_report"