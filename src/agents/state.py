"""
AlphaSignal Agent State
------------------------
The shared state dictionary that flows through every node
in the LangGraph pipeline.

Every agent reads from this state and writes back to it.
Think of it as the pipeline's shared memory — each agent
picks up where the last one left off.

TypedDict enforces the schema at development time,
helping catch bugs before they reach production.
"""

from typing import Optional
from typing_extensions import TypedDict


class AlphaSignalState(TypedDict):
    """
    Complete state schema for the AlphaSignal agent pipeline.

    Fields are grouped by which agent produces them:
        Input fields         → set by the user before the graph runs
        Fetcher fields       → set by Agent 1 (Document Fetcher)
        Extractor fields     → set by Agent 2 (Financial Extractor)
        Sentiment fields     → set by Agent 3 (Sentiment Analyzer)
        Checker fields       → set by Agent 4 (Self-Checker)
        Report fields        → set by Agent 5 (Report Writer)
        Monitoring fields    → updated by every agent
    """

    # ── Input (set before graph runs) ─────────────────────────────────────────
    ticker:             str               # e.g. "AAPL"
    filing_date:        Optional[str]     # e.g. "2024-11-01", None = latest

    # ── Agent 1: Document Fetcher ─────────────────────────────────────────────
    company_name:       Optional[str]     # "Apple Inc."
    raw_filing_path:    Optional[str]     # path to downloaded .txt file
    clean_text_path:    Optional[str]     # path to cleaned text
    sections_path:      Optional[str]     # path to sections JSON
    chunks_path:        Optional[str]     # path to chunks JSON
    cik:                Optional[str]     # SEC CIK number

    # ── Agent 2: Financial Extractor ──────────────────────────────────────────
    financials:         Optional[dict]    # structured financial metrics JSON
    financials_path:    Optional[str]     # path to saved financials JSON

    # ── Agent 3: Sentiment Analyzer ───────────────────────────────────────────
    sentiment:          Optional[dict]    # sentiment analysis results
    sentiment_path:     Optional[str]     # path to saved sentiment JSON

    # ── Agent 4: Self-Checker ─────────────────────────────────────────────────
    confidence_score:   Optional[float]  # combined pipeline confidence 0-1
    confidence_label:   Optional[str]    # "HIGH", "MEDIUM", "LOW"
    needs_recheck:      bool             # True = loop back to extractor
    recheck_count:      int              # how many times we've rechecked

    # ── Agent 5: Report Writer ────────────────────────────────────────────────
    report:             Optional[str]    # final markdown research report
    report_path:        Optional[str]    # path to saved report file

    # ── Monitoring (updated by every agent) ───────────────────────────────────
    errors:             list             # errors encountered during pipeline
    completed_steps:    list             # names of completed agent steps
    current_step:       Optional[str]    # which agent is currently running