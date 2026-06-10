"""
AlphaSignal Agent Nodes — Placeholders
-----------------------------------------
Each function here is one node in the LangGraph pipeline.
Today (Day 10) these are stubs that print their name and pass state through.
Days 11-15 replace each stub with the real implementation.

This pattern — stub first, implement later — is standard in
production engineering. It lets you verify the graph wiring
is correct before adding complex logic.
"""

from src.agents.state import AlphaSignalState


# ── Agent 1: Document Fetcher ─────────────────────────────────────────────────
def fetch_documents(state: AlphaSignalState) -> dict:
    """
    Fetches SEC 10-K filing for the given ticker.
    Parses HTML, cleans text, extracts sections, creates chunks,
    embeds into ChromaDB.

    Inputs:  state["ticker"], state["filing_date"]
    Outputs: state["raw_filing_path"], state["clean_text_path"],
             state["sections_path"], state["chunks_path"],
             state["company_name"], state["cik"]
    """
    print(f"  [Agent 1] Document Fetcher — ticker: {state['ticker']}")
    print(f"  [Agent 1] STUB — will be implemented Day 11")

    return {
        "completed_steps": state.get("completed_steps", []) + ["fetch_documents"],
        "current_step":    "extract_financials",
    }


# ── Agent 2: Financial Extractor ──────────────────────────────────────────────
def extract_financials(state: AlphaSignalState) -> dict:
    """
    Extracts structured financial metrics from the filing.
    Revenue, net income, EPS, operating expenses, product segments.
    Calculates year-over-year changes.

    Inputs:  state["ticker"], state["chunks_path"]
    Outputs: state["financials"], state["financials_path"]
    """
    print(f"  [Agent 2] Financial Extractor — ticker: {state['ticker']}")
    print(f"  [Agent 2] STUB — will be implemented Day 12")

    return {
        "completed_steps": state.get("completed_steps", []) + ["extract_financials"],
        "current_step":    "analyze_sentiment",
    }


# ── Agent 3: Sentiment Analyzer ───────────────────────────────────────────────
def analyze_sentiment(state: AlphaSignalState) -> dict:
    """
    Analyzes tone and language in the MD&A section.
    Loughran-McDonald lexicon + LLM dimensional analysis.

    Inputs:  state["ticker"]
    Outputs: state["sentiment"], state["sentiment_path"]
    """
    print(f"  [Agent 3] Sentiment Analyzer — ticker: {state['ticker']}")
    print(f"  [Agent 3] STUB — will be implemented Day 13")

    return {
        "completed_steps": state.get("completed_steps", []) + ["analyze_sentiment"],
        "current_step":    "check_confidence",
    }


# ── Agent 4: Self-Checker ─────────────────────────────────────────────────────
def check_confidence(state: AlphaSignalState) -> dict:
    """
    Evaluates the quality of extracted data.
    If confidence is too low, flags for re-extraction.
    Prevents low-quality data from reaching the report.

    Inputs:  state["financials"], state["sentiment"]
    Outputs: state["confidence_score"], state["confidence_label"],
             state["needs_recheck"]
    """
    print(f"  [Agent 4] Self-Checker — evaluating pipeline confidence")
    print(f"  [Agent 4] STUB — will be implemented Day 14")

    return {
        "confidence_score": 0.75,   # placeholder
        "confidence_label": "HIGH",
        "needs_recheck":    False,
        "recheck_count":    state.get("recheck_count", 0),
        "completed_steps":  state.get("completed_steps", []) + ["check_confidence"],
        "current_step":     "write_report",
    }


# ── Agent 5: Report Writer ────────────────────────────────────────────────────
def write_report(state: AlphaSignalState) -> dict:
    """
    Synthesizes all analysis into a structured research report.
    Combines financials, sentiment, and RAG Q&A into markdown.

    Inputs:  state["financials"], state["sentiment"],
             state["ticker"], state["company_name"]
    Outputs: state["report"], state["report_path"]
    """
    print(f"  [Agent 5] Report Writer — generating research report")
    print(f"  [Agent 5] STUB — will be implemented Day 15")

    return {
        "report":           f"[STUB REPORT] Analysis for {state['ticker']} — to be generated",
        "completed_steps":  state.get("completed_steps", []) + ["write_report"],
        "current_step":     "complete",
    }


# ── Routing Function ──────────────────────────────────────────────────────────
def route_after_confidence_check(state: AlphaSignalState) -> str:
    """
    Conditional routing logic for the Self-Checker node.

    This is the 'brain' of the self-correcting loop:
        - If confidence is HIGH or MEDIUM → proceed to report
        - If confidence is LOW and we haven't rechecked yet → loop back
        - If we've already rechecked twice → proceed anyway with warning

    Returns the name of the next node to route to.
    """
    needs_recheck = state.get("needs_recheck", False)
    recheck_count = state.get("recheck_count", 0)
    confidence    = state.get("confidence_score", 1.0)

    if needs_recheck and recheck_count < 2:
        print(f"  [Router] Confidence {confidence:.2f} too low — re-extracting (attempt {recheck_count + 1})")
        return "extract_financials"
    elif recheck_count >= 2:
        print(f"  [Router] Max rechecks reached — proceeding to report with warning")
        return "write_report"
    else:
        print(f"  [Router] Confidence {confidence:.2f} acceptable — proceeding to report")
        return "write_report"