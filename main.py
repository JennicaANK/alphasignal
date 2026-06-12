"""
AlphaSignal FastAPI Backend
-----------------------------
Production REST API wrapping the multi-agent pipeline.

Key design decisions:
    - Async background tasks: pipeline runs in background,
      API returns immediately with a job_id to poll.
      This prevents 3-minute HTTP timeouts.

    - In-memory job store: simple dict tracking job status.
      Production would use Redis or a database.

    - Caching: if results already exist for a ticker,
      return them instantly without re-running the pipeline.

    - Auto-docs: visit /docs for interactive API documentation.
      FastAPI generates this automatically from type hints — free.
"""

import uuid
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title        = "AlphaSignal API",
    description  = "Multi-agent financial intelligence system. Analyzes SEC 10-K filings autonomously.",
    version      = "1.0.0",
    docs_url     = "/docs",
    redoc_url    = "/redoc",
)

# CORS — allows the Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── In-memory job store ───────────────────────────────────────────────────────
# Tracks running and completed pipeline jobs.
# Format: { job_id: { status, ticker, started_at, completed_at, error } }
JOBS: dict = {}

# ── Pydantic models ───────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze"""
    ticker:      str
    filing_date: Optional[str] = None
    force_refresh: bool = False    # if True, re-run even if cached


class JobResponse(BaseModel):
    """Response from POST /analyze"""
    job_id:     str
    ticker:     str
    status:     str
    message:    str
    poll_url:   str


class StatusResponse(BaseModel):
    """Response from GET /status/{job_id}"""
    job_id:       str
    ticker:       str
    status:       str             # "running" | "complete" | "failed"
    started_at:   str
    completed_at: Optional[str]
    duration_sec: Optional[float]
    error:        Optional[str]


# ── Helper: check cached results ──────────────────────────────────────────────
def get_cached_results(ticker: str) -> Optional[dict]:
    """
    Check if we already have analysis results for this ticker.
    Returns the results dict if found, None otherwise.

    Caching prevents unnecessary re-runs when the filing hasn't changed.
    In production you'd also check if the filing date has changed.
    """
    pattern   = f"data/processed/{ticker}_*_financials.json"
    from glob import glob
    files = sorted(glob(pattern))

    if not files:
        return None

    latest = files[-1]
    try:
        with open(latest, "r") as f:
            financials = json.load(f)

        # Check for sentiment
        sent_pattern = f"data/processed/{ticker}_*_sentiment.json"
        sent_files   = sorted(glob(sent_pattern))
        sentiment    = {}
        if sent_files:
            with open(sent_files[-1], "r") as f:
                sentiment = json.load(f)

        # Check for report
        report_pattern = f"reports/{ticker}_*_report.md"
        report_files   = sorted(glob(report_pattern))
        report         = ""
        report_path    = ""
        if report_files:
            report_path = report_files[-1]
            with open(report_path, "r") as f:
                report = f.read()

        return {
            "ticker":     ticker,
            "financials": financials,
            "sentiment":  sentiment,
            "report":     report,
            "report_path": report_path,
            "cached":     True,
        }

    except Exception:
        return None


# ── Background pipeline runner ─────────────────────────────────────────────────
def run_pipeline(job_id: str, ticker: str, filing_date: Optional[str]):
    """
    Runs the full AlphaSignal multi-agent pipeline in the background.
    Updates the JOBS dict with status as it progresses.

    This runs in a separate thread via FastAPI's BackgroundTasks,
    so the API response is returned immediately while this runs.
    """
    JOBS[job_id]["status"] = "running"

    try:
        from src.agents.graph import build_graph

        pipeline = build_graph()

        initial_state = {
            "ticker":           ticker,
            "filing_date":      filing_date,
            "company_name":     None,
            "raw_filing_path":  None,
            "clean_text_path":  None,
            "sections_path":    None,
            "chunks_path":      None,
            "cik":              None,
            "financials":       None,
            "financials_path":  None,
            "sentiment":        None,
            "sentiment_path":   None,
            "confidence_score": None,
            "confidence_label": None,
            "needs_recheck":    False,
            "recheck_count":    0,
            "report":           None,
            "report_path":      None,
            "errors":           [],
            "completed_steps":  [],
            "current_step":     "fetch_documents",
        }

        final_state = pipeline.invoke(initial_state)

        # Mark job as complete
        completed_at = datetime.now().isoformat()
        started_at   = JOBS[job_id]["started_at"]
        duration     = (
            datetime.fromisoformat(completed_at) -
            datetime.fromisoformat(started_at)
        ).total_seconds()

        JOBS[job_id].update({
            "status":        "complete",
            "completed_at":  completed_at,
            "duration_sec":  round(duration, 1),
            "company_name":  final_state.get("company_name"),
            "filing_date":   final_state.get("filing_date"),
            "confidence":    final_state.get("confidence_score"),
            "report_path":   final_state.get("report_path"),
            "errors":        final_state.get("errors", []),
            "completed_steps": final_state.get("completed_steps", []),
        })

    except Exception as e:
        JOBS[job_id].update({
            "status": "failed",
            "error":  str(e),
            "completed_at": datetime.now().isoformat(),
        })


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    """Welcome endpoint — confirms API is running."""
    return {
        "name":        "AlphaSignal API",
        "version":     "1.0.0",
        "status":      "running",
        "description": "Multi-agent financial intelligence system",
        "endpoints": {
            "analyze":    "POST /analyze",
            "status":     "GET /status/{job_id}",
            "results":    "GET /results/{ticker}",
            "report":     "GET /report/{ticker}",
            "financials": "GET /financials/{ticker}",
            "docs":       "GET /docs",
        }
    }


@app.get("/health", tags=["General"])
def health():
    """Health check — used by deployment platforms to verify the app is alive."""
    return {
        "status":    "healthy",
        "timestamp": datetime.now().isoformat(),
        "jobs":      len(JOBS),
    }


@app.post("/analyze", response_model=JobResponse, status_code=202, tags=["Pipeline"])
def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Start the AlphaSignal pipeline for a ticker symbol.

    Returns immediately with a job_id.
    Poll GET /status/{job_id} to check progress.
    Retrieve results at GET /results/{ticker} when complete.

    HTTP 202 Accepted — the request is accepted but processing is async.
    """
    ticker = request.ticker.upper().strip()

    # Check cache first unless force_refresh
    if not request.force_refresh:
        cached = get_cached_results(ticker)
        if cached:
            return JSONResponse(
                status_code = 200,
                content     = {
                    "job_id":   "cached",
                    "ticker":   ticker,
                    "status":   "complete",
                    "message":  f"Returning cached results for {ticker}. Use force_refresh=true to re-run.",
                    "poll_url": f"/results/{ticker}",
                }
            )

    # Create job
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id":     job_id,
        "ticker":     ticker,
        "status":     "queued",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error":      None,
    }

    # Start pipeline in background
    background_tasks.add_task(
        run_pipeline,
        job_id      = job_id,
        ticker      = ticker,
        filing_date = request.filing_date,
    )

    return {
        "job_id":   job_id,
        "ticker":   ticker,
        "status":   "queued",
        "message":  f"Pipeline started for {ticker}. Poll /status/{job_id} for updates.",
        "poll_url": f"/status/{job_id}",
    }


@app.get("/status/{job_id}", response_model=StatusResponse, tags=["Pipeline"])
def get_status(job_id: str):
    """
    Check the status of a running pipeline job.

    Statuses:
        queued   → job is waiting to start
        running  → pipeline is executing
        complete → pipeline finished successfully
        failed   → pipeline encountered an error
    """
    if job_id not in JOBS:
        raise HTTPException(
            status_code = 404,
            detail      = f"Job '{job_id}' not found. It may have expired."
        )

    job = JOBS[job_id]
    return {
        "job_id":       job["job_id"],
        "ticker":       job["ticker"],
        "status":       job["status"],
        "started_at":   job["started_at"],
        "completed_at": job.get("completed_at"),
        "duration_sec": job.get("duration_sec"),
        "error":        job.get("error"),
    }


@app.get("/results/{ticker}", tags=["Results"])
def get_results(ticker: str):
    """
    Get the full analysis results for a ticker.
    Returns financials, sentiment, confidence, and report path.
    """
    ticker  = ticker.upper()
    cached  = get_cached_results(ticker)

    if not cached:
        raise HTTPException(
            status_code = 404,
            detail      = f"No results found for {ticker}. Run POST /analyze first."
        )

    # Return summary (not full report text — use /report endpoint for that)
    fin = cached.get("financials", {})
    inc = fin.get("income_statement", {})
    lex = cached.get("sentiment", {}).get("lexicon", {})

    return {
        "ticker":      ticker,
        "cached":      cached["cached"],
        "financials":  {
            "revenue_y1":    inc.get("total_net_sales", {}).get("year_1"),
            "net_income_y1": inc.get("net_income",      {}).get("year_1"),
            "years":         fin.get("years", {}),
        },
        "sentiment": {
            "label":          lex.get("sentiment_label"),
            "net_score":      lex.get("net_sentiment_score"),
            "overall_tone":   cached.get("sentiment", {}).get("llm_analysis", {}).get("overall_tone"),
        },
        "report_available": bool(cached.get("report")),
        "report_path":      cached.get("report_path"),
    }


@app.get("/report/{ticker}", tags=["Results"], response_class=PlainTextResponse)
def get_report(ticker: str):
    """
    Get the full markdown research report for a ticker.
    Returns plain text markdown.
    """
    ticker = ticker.upper()
    cached = get_cached_results(ticker)

    if not cached or not cached.get("report"):
        raise HTTPException(
            status_code = 404,
            detail      = f"No report found for {ticker}. Run POST /analyze first."
        )

    return cached["report"]


@app.get("/financials/{ticker}", tags=["Results"])
def get_financials(ticker: str):
    """
    Get structured financial metrics JSON for a ticker.
    Includes income statement, operating expenses, product segments, and YoY changes.
    """
    ticker = ticker.upper()
    cached = get_cached_results(ticker)

    if not cached or not cached.get("financials"):
        raise HTTPException(
            status_code = 404,
            detail      = f"No financial data found for {ticker}. Run POST /analyze first."
        )

    return cached["financials"]


@app.get("/jobs", tags=["General"])
def list_jobs():
    """List all pipeline jobs and their statuses."""
    return {
        "total": len(JOBS),
        "jobs":  [
            {
                "job_id":   j["job_id"],
                "ticker":   j["ticker"],
                "status":   j["status"],
                "started":  j["started_at"],
            }
            for j in JOBS.values()
        ]
    }