"""
Financial Data Extractor for AlphaSignal
------------------------------------------
Extracts structured financial metrics from 10-K filings
using targeted RAG queries and LLM-based JSON extraction.

This is the bridge between unstructured SEC filings
and clean, machine-readable financial data.

Output feeds:
    - Trend visualization dashboard (Week 4)
    - Automated PDF report generation (Week 4)
    - Multi-company comparison (Week 3)
    - Sentiment analysis (Day 8)

Key technique: Structured output extraction
    Instead of asking the LLM to explain, we instruct it
    to respond ONLY in valid JSON matching our schema.
    Temperature = 0.0 for maximum determinism.
"""

import json
import os
import re
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from src.rag.vector_store import query_store
from src.rag.reranker import rerank

load_dotenv()

GROQ_MODEL  = "llama-3.1-8b-instant"
TEMPERATURE = 0.0    # zero temperature = most deterministic JSON output


# ── Financial Schema ──────────────────────────────────────────────────────────
# Defines exactly what we want to extract.
# null values get filled by the LLM.
EXTRACTION_SCHEMA = {
    "company":      None,
    "ticker":       None,
    "filing_date":  None,
    "currency":     "USD millions",
    "income_statement": {
        "total_net_sales":  {"year_1": None, "year_2": None, "year_3": None},
        "gross_margin":     {"year_1": None, "year_2": None, "year_3": None},
        "operating_income": {"year_1": None, "year_2": None, "year_3": None},
        "net_income":       {"year_1": None, "year_2": None, "year_3": None},
        "eps_basic":        {"year_1": None, "year_2": None, "year_3": None},
        "eps_diluted":      {"year_1": None, "year_2": None, "year_3": None},
    },
    "operating_expenses": {
        "research_and_development":      {"year_1": None, "year_2": None, "year_3": None},
        "selling_general_admin":         {"year_1": None, "year_2": None, "year_3": None},
        "total_operating_expenses":      {"year_1": None, "year_2": None, "year_3": None},
    },
    "product_segments": {
        "iphone":                  {"year_1": None, "year_2": None, "year_3": None},
        "mac":                     {"year_1": None, "year_2": None, "year_3": None},
        "ipad":                    {"year_1": None, "year_2": None, "year_3": None},
        "services":                {"year_1": None, "year_2": None, "year_3": None},
        "wearables_home_acc":      {"year_1": None, "year_2": None, "year_3": None},
    }
}


# ── Targeted RAG Queries ──────────────────────────────────────────────────────
# Each query is designed to retrieve chunks covering one part of the financials.
# Multiple queries cast a wider net than one general query.
FINANCIAL_QUERIES = [
    "total net sales revenue income statement consolidated",
    "gross margin operating income net income earnings",
    "earnings per share basic diluted EPS",
    "operating expenses research development selling general administrative",
    "iPhone Mac iPad Services Wearables net sales product segments",
    "iPhone revenue net sales annual",          # new
    "diluted basic earnings per share annual",
    "net income per share diluted basic weighted average shares",   # new
]


# ── Context Collector ─────────────────────────────────────────────────────────
def collect_financial_context(
    ticker:    str,
    n_per_query: int = 3
) -> str:
    """
    Run multiple targeted RAG queries and collect unique chunks.

    Why multiple queries?
        A single query like "financial data" is too broad.
        Specific queries like "earnings per share diluted"
        retrieve the exact table rows we need.

    Deduplication by chunk ID ensures the same chunk
    doesn't appear multiple times in the context.
    """
    print(f"  Running {len(FINANCIAL_QUERIES)} targeted queries...")

    seen_ids = set()
    all_chunks = []

    for query in FINANCIAL_QUERIES:
        raw_chunks = query_store(query, n_results=n_per_query, ticker=ticker)
        reranked   = rerank(query, raw_chunks, top_k=2)

        for chunk in reranked:
            chunk_id = chunk["metadata"].get("chunk_index", "")
            section  = chunk["metadata"].get("section", "")
            unique   = f"{section}_{chunk_id}"

            if unique not in seen_ids:
                seen_ids.add(unique)
                all_chunks.append(chunk)

    print(f"  Collected {len(all_chunks)} unique chunks across all queries")

    # Build context string
    context_parts = []
    for i, chunk in enumerate(all_chunks):
        section = chunk["metadata"].get("section", "unknown")
        context_parts.append(f"[Chunk {i+1} | {section}]\n{chunk['text']}")

    full_context = "\n\n---\n\n".join(context_parts)

    return full_context[:12000] # cap at 12000 tokens


# ── Extraction Prompt ─────────────────────────────────────────────────────────
def build_extraction_prompt(context: str, ticker: str) -> tuple[str, str]:
    """
    Build the structured extraction prompt.

    Key technique: instruct the LLM to output ONLY valid JSON.
    No preamble. No explanation. No markdown code blocks.
    Just raw JSON matching our schema.

    This is called 'structured output extraction' and is
    the standard way to get machine-readable data from LLMs.
    """
    schema_str = json.dumps(EXTRACTION_SCHEMA, indent=2)

    system_message = f"""You are a financial data extraction engine.
Your ONLY job is to extract financial figures from the provided context
and return them as valid JSON.

CRITICAL RULES:
1. Output ONLY valid JSON. No explanation, no preamble, no markdown.
2. Do not wrap in ```json blocks. Return raw JSON only.
3. Use null for ANY value you are not 100% certain about from the context. NEVER guess, estimate, or approximate. If the exact figure is not clearly visible in the context, return null.
4. NEVER return negative values for revenue or segment sales. If a value seems negative, return null instead.
5. All monetary values must be numbers in millions (no $ signs, no commas).
6. EPS values are per-share amounts (e.g. 6.42 not 6,420).
7. year_1 = most recent year, year_2 = prior year, year_3 = two years ago.
8. Include the actual year labels in a "years" field.

Return exactly this JSON structure with values filled in:
{schema_str}

Add a "years" field at the top level: {{"year_1": "YYYY", "year_2": "YYYY", "year_3": "YYYY"}}"""

    user_message = f"""Extract all financial metrics from this {ticker} 10-K context:

{context}

Return ONLY valid JSON. No other text."""

    return system_message, user_message


# ── JSON Parser ───────────────────────────────────────────────────────────────
def parse_llm_json(raw_response: str) -> dict:
    """
    Parse JSON from LLM response robustly.

    LLMs sometimes wrap JSON in markdown code blocks even
    when told not to. This function handles both cases:
        - Raw JSON: { "company": "Apple"... }
        - Wrapped:  ```json\n{ "company": "Apple"... }\n```
    """
    # Strip markdown code fences if present
    text = raw_response.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*',     '', text)
    text = re.sub(r'\s*```$',     '', text)
    text = text.strip()

    # Find JSON boundaries in case there's surrounding text
    start = text.find('{')
    end   = text.rfind('}') + 1

    if start == -1 or end == 0:
        raise ValueError("No JSON object found in LLM response")

    json_str = text[start:end]
    return json.loads(json_str)


# ── YoY Change Calculator ─────────────────────────────────────────────────────
def calculate_yoy_changes(data: dict) -> dict:
    """
    Add year-over-year percentage changes for every metric.

    Formula: ((year_1 - year_2) / abs(year_2)) * 100

    This is one of the most valuable outputs for financial analysis —
    raw numbers matter less than how they changed.
    """
    yoy = {}

    def pct_change(new, old):
        if new is None or old is None:
            return None
        if old == 0:
            return None
        return round(((new - old) / abs(old)) * 100, 2)

    for category in ["income_statement", "operating_expenses", "product_segments"]:
        if category not in data:
            continue
        yoy[category] = {}
        for metric, values in data[category].items():
            if isinstance(values, dict):
                y1 = values.get("year_1")
                y2 = values.get("year_2")
                yoy[category][metric] = {
                    "yoy_change_pct":    pct_change(y1, y2),
                    "yoy_change_abs":    round(y1 - y2, 2) if y1 and y2 else None,
                    "direction":         "▲" if y1 and y2 and y1 > y2 else "▼"
                }

    data["yoy_changes"] = yoy
    return data


# ── Validation ────────────────────────────────────────────────────────────────
def validate_extraction(data: dict) -> list[str]:
    """
    Basic sanity checks on extracted data.
    Returns a list of warnings for any suspicious values.

    In production you'd compare against known benchmarks.
    Here we check for basic reasonableness.
    """
    warnings = []

    try:
        revenue = data["income_statement"]["total_net_sales"]["year_1"]
        if revenue and revenue < 1000:
            warnings.append(f"Revenue {revenue} seems too low — expected billions")
        if revenue and revenue > 10_000_000:
            warnings.append(f"Revenue {revenue} seems too high — check units")
    except (KeyError, TypeError):
        warnings.append("Could not validate revenue figure")

    try:
        eps = data["income_statement"]["eps_diluted"]["year_1"]
        if eps and eps > 1000:
            warnings.append(f"EPS {eps} seems too high — should be per-share amount")
    except (KeyError, TypeError):
        pass

    return warnings


# ── Unit Fix ─────────────────────────────────────────────────────────────────
def fix_units(data: dict) -> dict:
    """
    Auto-detect and fix unit mismatch.
    If the LLM returned values in billions instead of millions,
    multiply all numeric values by 1000 to normalize to millions.
    Revenue for large companies should be > 10,000 (millions).
    If it's < 10,000 it's likely in billions — multiply by 1000.
    """
    try:
        revenue = data["income_statement"]["total_net_sales"]["year_1"]
        if revenue and isinstance(revenue, (int, float)) and revenue < 10000:
            print(f"  Unit fix: values appear to be in billions, converting to millions...")
            multiplier = 1000

            for category in ["income_statement", "operating_expenses", "product_segments"]:
                if category not in data:
                    continue
                for metric, values in data[category].items():
                    if not isinstance(values, dict):
                        continue
                    for year_key in ["year_1", "year_2", "year_3"]:
                        val = values.get(year_key)
                        # Only multiply large monetary values, not EPS
                        if val and isinstance(val, (int, float)) and abs(val) > 0.01:
                            if metric not in ["eps_basic", "eps_diluted"]:
                                values[year_key] = round(val * multiplier, 0)
    except (KeyError, TypeError):
        pass
    return data

# ── Invalid Value Cleanup ─────────────────────────────────────────────────────

def clean_invalid_values(data: dict) -> dict:
    """
    Replace negative or near-zero values in monetary metrics with None.
    Revenue and segment sales are always positive.
    Negative values mean the LLM couldn't find the data and guessed.
    EPS can legitimately be small but not negative for Apple.
    """
    monetary_categories = [
        "income_statement",
        "operating_expenses",
        "product_segments"
    ]
    non_negative_metrics = [
        "total_net_sales", "gross_margin", "operating_income",
        "net_income", "research_and_development", "selling_general_admin",
        "total_operating_expenses", "iphone", "mac", "ipad",
        "services", "wearables_home_acc"
    ]

    for category in monetary_categories:
        if category not in data:
            continue
        for metric, values in data[category].items():
            if not isinstance(values, dict):
                continue
            if metric not in non_negative_metrics:
                continue
            for year_key in ["year_1", "year_2", "year_3"]:
                val = values.get(year_key)
                if val is not None and isinstance(val, (int, float)):
                    # Flag clearly wrong values: negative or suspiciously small
                    if val < 0 or (val < 100 and metric != "eps_diluted" and metric != "eps_basic"):
                        values[year_key] = None

    return data

# ── Main Extractor ────────────────────────────────────────────────────────────
def extract_financials(
    ticker:      str,
    filing_date: str = None,
    output_dir:  str = "data/processed"
) -> dict:
    """
    Master function — full extraction pipeline.

    1. Collect context via targeted RAG queries
    2. Build structured extraction prompt
    3. LLM extracts JSON
    4. Parse and validate
    5. Calculate YoY changes
    6. Save to disk
    """
    print(f"\n{'='*55}")
    print(f"AlphaSignal — Financial Extractor: {ticker}")
    print(f"{'='*55}")

    # Step 1: Collect context
    print("\nStep 1: Collecting financial context...")
    context = collect_financial_context(ticker)

    # Step 2: Build prompt
    print("Step 2: Building extraction prompt...")
    system_msg, user_msg = build_extraction_prompt(context, ticker)
    total_chars = len(system_msg) + len(user_msg)
    print(f"  Prompt size: {total_chars:,} characters (~{total_chars//4:,} tokens)")

    # Step 3: LLM extraction
    print("Step 3: Extracting structured data via Groq...")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=TEMPERATURE,
        max_tokens=1200,
    )
    raw_output = response.choices[0].message.content

    # Step 4: Parse JSON
    print("Step 4: Parsing JSON response...")
    try:
        data = parse_llm_json(raw_output)
        print("  JSON parsed successfully")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  JSON parse failed: {e}")
        print(f"  Raw response: {raw_output[:500]}")
        return {}

    # Add ticker if not extracted
    if not data.get("ticker"):
        data["ticker"] = ticker

    # Step 5: Validate
    print("Step 5: Validating extracted values...")
    data = fix_units(data)
    data = clean_invalid_values(data)
    warnings = validate_extraction(data)
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("  All values look reasonable")

    # Step 6: YoY changes
    print("Step 6: Calculating year-over-year changes...")
    data = calculate_yoy_changes(data)

    # Step 7: Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    date_str = filing_date or "latest"
    out_path = os.path.join(output_dir, f"{ticker}_{date_str}_financials.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to: {out_path}")

    return data


# ── Pretty Printer ────────────────────────────────────────────────────────────
def print_financials(data: dict) -> None:
    """Print extracted financials in a readable table format."""
    if not data:
        print("No data to display.")
        return

    years = data.get("years", {})
    y1 = years.get("year_1", "Y1")
    y2 = years.get("year_2", "Y2")
    y3 = years.get("year_3", "Y3")

    print(f"\n{'='*65}")
    print(f"  {data.get('company', 'Unknown')} | {data.get('ticker')} | {data.get('currency', 'USD millions')}")
    print(f"{'='*65}")
    print(f"  {'Metric':<35} {y1:>10} {y2:>10} {y3:>10}  YoY%")
    print(f"  {'─'*63}")

    def row(label, category, metric):
        try:
            vals = data[category][metric]
            v1   = vals.get("year_1")
            v2   = vals.get("year_2")
            v3   = vals.get("year_3")
            yoy  = data.get("yoy_changes", {}).get(category, {}).get(metric, {})
            pct  = yoy.get("yoy_change_pct")
            dirn = yoy.get("direction", "")
            v1s  = f"{v1:>10,.0f}" if isinstance(v1, (int, float)) else f"{'N/A':>10}"
            v2s  = f"{v2:>10,.0f}" if isinstance(v2, (int, float)) else f"{'N/A':>10}"
            v3s  = f"{v3:>10,.0f}" if isinstance(v3, (int, float)) else f"{'N/A':>10}"
            pcts = f"  {dirn}{pct:+.1f}%" if pct is not None else ""
            print(f"  {label:<35} {v1s} {v2s} {v3s}{pcts}")
        except (KeyError, TypeError):
            print(f"  {label:<35} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    print(f"\n  INCOME STATEMENT")
    row("Total Net Sales",             "income_statement", "total_net_sales")
    row("Gross Margin",                "income_statement", "gross_margin")
    row("Operating Income",            "income_statement", "operating_income")
    row("Net Income",                  "income_statement", "net_income")
    row("EPS (Diluted)",               "income_statement", "eps_diluted")

    print(f"\n  OPERATING EXPENSES")
    row("R&D",                         "operating_expenses", "research_and_development")
    row("SG&A",                        "operating_expenses", "selling_general_admin")
    row("Total OpEx",                  "operating_expenses", "total_operating_expenses")

    print(f"\n  PRODUCT SEGMENTS")
    row("iPhone",                      "product_segments", "iphone")
    row("Mac",                         "product_segments", "mac")
    row("iPad",                        "product_segments", "ipad")
    row("Services",                    "product_segments", "services")
    row("Wearables & Home",            "product_segments", "wearables_home_acc")

    print(f"{'='*65}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = extract_financials(ticker="AAPL", filing_date="2024-11-01")
    print_financials(data)