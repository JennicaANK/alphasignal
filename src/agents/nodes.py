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
    Connects sentiment_analyzer.py into the agent pipeline.

    Inputs:  state["ticker"], state["filing_date"]
    Outputs: state["sentiment"], state["sentiment_path"]
    """
    ticker      = state.get("ticker")
    filing_date = state.get("filing_date")

    print(f"\n  [Agent 3] Sentiment Analyzer — {ticker}")

    try:
        from src.utils.sentiment_analyzer import analyze_sentiment as run_sentiment

        sentiment = run_sentiment(
            ticker      = ticker,
            filing_date = filing_date,
        )

        if not sentiment:
            raise ValueError("Sentiment analysis returned empty result")

        lex   = sentiment.get("lexicon", {})
        llm   = sentiment.get("llm_analysis", {})

        print(f"  [Agent 3] Sentiment:        {lex.get('sentiment_label', 'N/A')}")
        print(f"  [Agent 3] Net score:        {lex.get('net_sentiment_score', 'N/A')}")
        print(f"  [Agent 3] Tone:             {llm.get('overall_tone', 'N/A')}")
        print(f"  [Agent 3] Tone score:       {llm.get('tone_score', 'N/A')}/10")
        print(f"  [Agent 3] Fwd confidence:   {llm.get('forward_confidence', 'N/A')}/10")

        sentiment_path = f"data/processed/{ticker}_{filing_date}_sentiment.json"

        return {
            "sentiment":       sentiment,
            "sentiment_path":  sentiment_path,
            "completed_steps": state.get("completed_steps", []) + ["analyze_sentiment"],
            "current_step":    "check_confidence",
            "errors":          state.get("errors", []),
        }

    except Exception as e:
        error_msg = f"analyze_sentiment failed: {str(e)}"
        print(f"  [Agent 3] ERROR: {error_msg}")

        return {
            "sentiment":       None,
            "errors":          state.get("errors", []) + [error_msg],
            "completed_steps": state.get("completed_steps", []) + ["analyze_sentiment_error"],
            "current_step":    "check_confidence",
        }


# ── Agent 4: Self-Checker ─────────────────────────────────────────────────────
def check_confidence(state: AlphaSignalState) -> dict:
    """
    Evaluates the quality of the entire pipeline so far.
    Computes a combined confidence score from two signals:

        Financial completeness (70% weight)
            How many key metrics were successfully extracted?
            Null values mean the LLM couldn't find the data.

        Sentiment quality (30% weight)
            Did the LLM dimensional analysis succeed?
            Did it produce a meaningful tone score?

    If combined confidence < 0.50 and recheck_count < 2:
        → sets needs_recheck = True
        → pipeline loops back to extract_financials

    This is uncertainty quantification at the pipeline level —
    not just on individual answers but on the entire analysis.
    """
    ticker        = state.get("ticker")
    financials    = state.get("financials", {})
    sentiment     = state.get("sentiment", {})
    recheck_count = state.get("recheck_count", 0)
    errors        = state.get("errors", [])

    print(f"\n  [Agent 4] Self-Checker — evaluating pipeline quality")

    # ── Score 1: Financial Completeness ───────────────────────────────────────
    # Check how many of our key metrics were successfully extracted
    key_metrics = [
        ("income_statement",    "total_net_sales"),
        ("income_statement",    "net_income"),
        ("income_statement",    "gross_margin"),
        ("income_statement",    "operating_income"),
        ("operating_expenses",  "research_and_development"),
        ("operating_expenses",  "total_operating_expenses"),
        ("product_segments",    "iphone"),
        ("product_segments",    "services"),
    ]

    filled  = 0
    total   = len(key_metrics)

    for category, metric in key_metrics:
        try:
            val = financials.get(category, {}).get(metric, {}).get("year_1")
            if val is not None and isinstance(val, (int, float)) and val > 0:
                filled += 1
        except (AttributeError, TypeError):
            pass

    financial_completeness = round(filled / total, 4) if total > 0 else 0.0
    print(f"  [Agent 4] Financial completeness: {filled}/{total} key metrics ({financial_completeness:.0%})")

    # ── Score 2: Sentiment Quality ────────────────────────────────────────────
    sentiment_quality = 0.0
    if sentiment:
        llm_result  = sentiment.get("llm_analysis", {})
        tone_score  = llm_result.get("tone_score")
        tone_label  = llm_result.get("overall_tone", "UNKNOWN")
        lex_count   = sentiment.get("lexicon", {}).get("word_count", 0)

        tone_ok     = tone_score is not None and tone_label != "UNKNOWN"
        text_ok     = lex_count > 100

        if tone_ok and text_ok:
            sentiment_quality = 1.0
        elif tone_ok or text_ok:
            sentiment_quality = 0.6
        else:
            sentiment_quality = 0.2

    print(f"  [Agent 4] Sentiment quality:      {sentiment_quality:.0%}")

    # ── Score 3: Error penalty ────────────────────────────────────────────────
    error_penalty = min(len(errors) * 0.1, 0.3)
    if error_penalty > 0:
        print(f"  [Agent 4] Error penalty:          -{error_penalty:.0%} ({len(errors)} errors)")

    # ── Combined confidence ───────────────────────────────────────────────────
    raw_confidence = (
        (0.70 * financial_completeness) +
        (0.30 * sentiment_quality)
    ) - error_penalty

    confidence_score = round(max(0.0, min(1.0, raw_confidence)), 4)

    # ── Confidence label ──────────────────────────────────────────────────────
    if confidence_score >= 0.70:
        label = "HIGH"
    elif confidence_score >= 0.50:
        label = "MEDIUM"
    else:
        label = "LOW"

    print(f"  [Agent 4] Combined confidence:    {confidence_score:.4f} [{label}]")

    # ── Routing decision ──────────────────────────────────────────────────────
    needs_recheck = False

    if confidence_score < 0.50 and recheck_count < 2:
        needs_recheck  = True
        recheck_count += 1
        print(f"  [Agent 4] Decision: RE-EXTRACT (confidence too low, attempt {recheck_count})")
    elif recheck_count >= 2:
        print(f"  [Agent 4] Decision: PROCEED (max rechecks reached — reporting with warning)")
    else:
        print(f"  [Agent 4] Decision: PROCEED (confidence acceptable)")

    return {
        "confidence_score": confidence_score,
        "confidence_label": label,
        "needs_recheck":    needs_recheck,
        "recheck_count":    recheck_count,
        "completed_steps":  state.get("completed_steps", []) + ["check_confidence"],
        "current_step":     "write_report",
        "errors":           errors,
    }



# ── Agent 5: Report Writer ────────────────────────────────────────────────────
def write_report(state: AlphaSignalState) -> dict:
    """
    Synthesizes all pipeline outputs into a professional research report.

    Combines:
        - Structured financial metrics from Agent 2
        - Sentiment analysis from Agent 3
        - Confidence scores from Agent 4
        - 3 targeted RAG answers (risks, strategy, outlook)

    Outputs a formatted markdown report saved to reports/
    """
    ticker         = state.get("ticker")
    company        = state.get("company_name", ticker)
    filing_date    = state.get("filing_date", "unknown")
    financials     = state.get("financials",  {})
    sentiment      = state.get("sentiment",   {})
    confidence     = state.get("confidence_score", 0.0)
    conf_label     = state.get("confidence_label",  "UNKNOWN")
    errors         = state.get("errors", [])

    print(f"\n  [Agent 5] Report Writer — generating report for {company}")

    try:
        import os
        from pathlib import Path
        from groq import Groq
        from dotenv import load_dotenv
        from src.rag.rag_pipeline import query_rag

        load_dotenv()

        # ── Step 1: Pull key financial figures ────────────────────────────────
        inc  = financials.get("income_statement",   {})
        opex = financials.get("operating_expenses", {})
        seg  = financials.get("product_segments",   {})
        yoy  = financials.get("yoy_changes",        {})
        yrs  = financials.get("years", {"year_1": "2025", "year_2": "2024", "year_3": "2023"})

        def val(category, metric, year="year_1"):
            try:
                v = financials.get(category, {}).get(metric, {}).get(year)
                return f"{v:,.0f}" if isinstance(v, (int, float)) else "N/A"
            except Exception:
                return "N/A"

        def pct(category, metric):
            try:
                p = yoy.get(category, {}).get(metric, {}).get("yoy_change_pct")
                d = yoy.get(category, {}).get(metric, {}).get("direction", "")
                return f"{d}{p:+.1f}%" if p is not None else "N/A"
            except Exception:
                return "N/A"

        y1 = yrs.get("year_1", "2025")
        y2 = yrs.get("year_2", "2024")
        y3 = yrs.get("year_3", "2023")

        # ── Step 2: Pull sentiment signals ────────────────────────────────────
        lex       = sentiment.get("lexicon",      {})
        llm_sent  = sentiment.get("llm_analysis", {})
        phrases   = sentiment.get("phrases",      {})

        tone        = llm_sent.get("overall_tone",       "N/A")
        tone_score  = llm_sent.get("tone_score",          "N/A")
        fwd_conf    = llm_sent.get("forward_confidence",  "N/A")
        pos_themes  = llm_sent.get("key_positive_themes", [])
        concerns    = llm_sent.get("key_concerns",        [])
        notable     = llm_sent.get("notable_language",    "")
        net_score   = lex.get("net_sentiment_score",      "N/A")

        # ── Step 3: Targeted RAG questions ────────────────────────────────────
        print(f"  [Agent 5] Running targeted RAG queries...")

        rag_questions = [
            "What are the top 3 risk factors for this company?",
            "What is the company's strategy and competitive advantages?",
            "What is management's outlook for the next fiscal year?",
        ]

        rag_answers = {}
        for q in rag_questions:
            result = query_rag(q, ticker=ticker, verbose=False)
            rag_answers[q] = {
                "answer":     result.get("answer", "Not found"),
                "confidence": result.get("confidence", {}).get("confidence", 0),
            }
            print(f"  [Agent 5] RAG: '{q[:50]}...' → confidence {rag_answers[q]['confidence']:.2f}")

        # ── Step 4: Generate report with LLM ─────────────────────────────────
        print(f"  [Agent 5] Generating report via Groq...")

        fin_summary = f"""
FINANCIAL METRICS (USD millions):
Revenue:        {val('income_statement','total_net_sales')} ({y1}) | {val('income_statement','total_net_sales','year_2')} ({y2}) | YoY {pct('income_statement','total_net_sales')}
Gross Margin:   {val('income_statement','gross_margin')}    ({y1}) | {val('income_statement','gross_margin','year_2')}    ({y2}) | YoY {pct('income_statement','gross_margin')}
Net Income:     {val('income_statement','net_income')}      ({y1}) | {val('income_statement','net_income','year_2')}      ({y2}) | YoY {pct('income_statement','net_income')}
R&D:            {val('operating_expenses','research_and_development')} ({y1}) | YoY {pct('operating_expenses','research_and_development')}
iPhone:         {val('product_segments','iphone')}    ({y1}) | YoY {pct('product_segments','iphone')}
Services:       {val('product_segments','services')}  ({y1}) | YoY {pct('product_segments','services')}
Mac:            {val('product_segments','mac')}       ({y1}) | YoY {pct('product_segments','mac')}"""

        sent_summary = f"""
SENTIMENT ANALYSIS:
Overall tone:       {tone}
Tone score:         {tone_score}/10
Forward confidence: {fwd_conf}/10
Net sentiment:      {net_score}
Key positives:      {', '.join(pos_themes) if pos_themes else 'None identified'}
Key concerns:       {', '.join(concerns)   if concerns   else 'None identified'}
Notable language:   {notable}"""

        rag_summary = "\n".join([
            f"Q: {q}\nA: {v['answer'][:300]}\nConfidence: {v['confidence']:.2f}\n"
            for q, v in rag_answers.items()
        ])

        system_msg = f"""You are a senior financial analyst writing a professional investment research report.
Write a structured markdown report based on the provided data.
Be factual, precise, and analytical. Use the exact numbers provided.
Format: use ## headers, bullet points for lists, bold for key figures."""

        user_msg = f"""Write a professional research report for {company} ({ticker}).

{fin_summary}

{sent_summary}

RAG ANALYSIS:
{rag_summary}

Pipeline confidence: {confidence:.4f} [{conf_label}]
Filing date: {filing_date}

Structure the report with these sections:
1. Executive Summary (3-4 sentences with the most important findings)
2. Financial Performance (key metrics with YoY analysis)
3. Segment Analysis (breakdown by product/service)
4. Management Tone & Sentiment
5. Key Risks
6. Investment Signals
7. Data Quality Note: State that pipeline confidence is {confidence:.4f} [{conf_label}] based on data completeness scoring. Do not reference individual RAG query scores here.

Keep it concise and professional. Maximum 600 words."""

        client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model       = "llama-3.1-8b-instant",
            messages    = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature = 0.2,
            max_tokens  = 1200,
        )

        report_body = response.choices[0].message.content.strip()

        # ── Step 5: Assemble final report ─────────────────────────────────────
        report = f"""# AlphaSignal Research Report: {company} ({ticker})

**Generated by AlphaSignal Multi-Agent Pipeline**
**Filing:** {filing_date} | **Pipeline Confidence:** {confidence:.4f} [{conf_label}]
**Errors:** {len(errors)} | **Agents completed:** {len(state.get('completed_steps', []))}

---

{report_body}

---
*Report generated autonomously by AlphaSignal — a multi-agent financial intelligence system.*
*Data source: SEC EDGAR 10-K filing. LLM: Groq Llama-3.1-8b-instant.*
"""

        # ── Step 6: Save report ───────────────────────────────────────────────
        Path("reports").mkdir(exist_ok=True)
        report_path = f"reports/{ticker}_{filing_date}_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"  [Agent 5] Report saved to: {report_path}")
        print(f"  [Agent 5] Report length:   {len(report):,} characters")

        return {
            "report":          report,
            "report_path":     report_path,
            "completed_steps": state.get("completed_steps", []) + ["write_report"],
            "current_step":    "complete",
            "errors":          errors,
        }

    except Exception as e:
        error_msg = f"write_report failed: {str(e)}"
        print(f"  [Agent 5] ERROR: {error_msg}")
        return {
            "report":          f"Report generation failed: {error_msg}",
            "errors":          state.get("errors", []) + [error_msg],
            "completed_steps": state.get("completed_steps", []) + ["write_report_error"],
            "current_step":    "complete",
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