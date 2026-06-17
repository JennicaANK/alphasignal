"""
AlphaSignal Streamlit Dashboard
---------------------------------
Interactive UI for the multi-agent financial intelligence system.
Users enter a ticker, the pipeline runs, results display as charts and tables.
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AlphaSignal",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

API_URL = "http://localhost:8000"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.3rem 0;
    }
    .metric-value { font-size: 1.6rem; font-weight: 600; color: #cdd6f4; }
    .metric-label { font-size: 0.8rem; color: #6c7086; text-transform: uppercase; letter-spacing: 0.05em; }
    .positive     { color: #a6e3a1 !important; }
    .negative     { color: #f38ba8 !important; }
    .neutral      { color: #fab387 !important; }
    .confidence-high   { color: #a6e3a1; }
    .confidence-medium { color: #fab387; }
    .confidence-low    { color: #f38ba8; }
    h1 { color: #cdd6f4 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_cached(ticker):
    try:
        r = requests.get(f"{API_URL}/results/{ticker}", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_financials(ticker):
    try:
        r = requests.get(f"{API_URL}/financials/{ticker}", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_report(ticker):
    try:
        r = requests.get(f"{API_URL}/report/{ticker}", timeout=5)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None


def run_pipeline(ticker, force=False):
    try:
        r = requests.post(
            f"{API_URL}/analyze",
            json    = {"ticker": ticker, "force_refresh": force},
            timeout = 10,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def poll_status(job_id, placeholder):
    for _ in range(180):   # max 3 minutes
        try:
            r = requests.get(f"{API_URL}/status/{job_id}", timeout=5)
            job = r.json()
            status = job.get("status")

            if status == "complete":
                placeholder.success(f"Pipeline complete in {job.get('duration_sec')}s")
                return True
            elif status == "failed":
                placeholder.error(f"Pipeline failed: {job.get('error')}")
                return False
            else:
                placeholder.info(f"Pipeline running... (this takes 2-3 minutes)")
                time.sleep(5)
        except Exception:
            time.sleep(5)
    return False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 AlphaSignal")
    st.markdown("*Multi-agent financial intelligence*")
    st.divider()

    # API status
    api_ok = check_api()
    if api_ok:
        st.success("API connected")
    else:
        st.error("API offline — run: uvicorn main:app --reload")

    st.divider()
    st.markdown("### Analyze a Company")

    ticker_input = st.text_input(
        "Ticker Symbol",
        value       = "AAPL",
        max_chars   = 5,
        placeholder = "e.g. AAPL, NVDA, MSFT",
    ).upper().strip()

    force_refresh = st.checkbox("Force re-analysis", value=False)

    analyze_btn = st.button(
        "Run AlphaSignal Pipeline",
        type     = "primary",
        disabled = not api_ok,
        use_container_width = True,
    )

    st.divider()
    st.markdown("#### How it works")
    st.markdown("""
1. Downloads 10-K from SEC EDGAR
2. Extracts financial metrics
3. Analyzes management sentiment
4. Self-checks data quality
5. Generates research report
    """)


# ── Main content ──────────────────────────────────────────────────────────────
st.title("📈 AlphaSignal")
st.markdown("*Autonomous financial intelligence powered by multi-agent AI*")
st.divider()

# Handle analyze button
if analyze_btn and ticker_input:
    status_placeholder = st.empty()
    result = run_pipeline(ticker_input, force=force_refresh)

    if result.get("status") == "complete":
        status_placeholder.success(f"Using cached results for {ticker_input}")
    elif result.get("job_id"):
        job_id = result["job_id"]
        if not poll_status(job_id, status_placeholder):
            st.stop()
    else:
        st.error(f"Failed to start pipeline: {result}")
        st.stop()

# Load results
results    = get_cached(ticker_input)
financials = get_financials(ticker_input) if results else None
report     = get_report(ticker_input)     if results else None

if not results:
    st.info(f"No data for **{ticker_input}** yet. Click **Run AlphaSignal Pipeline** to analyze.")
    st.stop()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.subheader(f"📊 {ticker_input} — Financial Overview")

fin = results.get("financials", {})
sen = results.get("sentiment",  {})

col1, col2, col3, col4 = st.columns(4)

with col1:
    rev = fin.get("revenue_y1")
    st.metric(
        "Total Revenue",
        f"${rev/1000:.1f}B" if rev else "N/A",
        label_visibility = "visible"
    )

with col2:
    ni = fin.get("net_income_y1")
    st.metric("Net Income", f"${ni/1000:.1f}B" if ni else "N/A")

with col3:
    tone = sen.get("overall_tone", "N/A")
    st.metric("Management Tone", tone)

with col4:
    yrs = fin.get("years", {})
    st.metric("Filing Year", yrs.get("year_1", "N/A"))

st.divider()

# ── Charts + Report Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Financial Charts", "🧠 Sentiment Analysis", "📄 Research Report"])

with tab1:
    if not financials:
        st.info("Run the pipeline to see charts.")
    else:
        inc  = financials.get("income_statement",   {})
        opex = financials.get("operating_expenses", {})
        seg  = financials.get("product_segments",   {})
        yrs  = financials.get("years", {"year_1": "2025", "year_2": "2024", "year_3": "2023"})
        years = [yrs.get("year_3"), yrs.get("year_2"), yrs.get("year_1")]

        def extract_series(data, metric):
            vals = data.get(metric, {})
            return [
                vals.get("year_3"),
                vals.get("year_2"),
                vals.get("year_1"),
            ]

        # Revenue & Net Income trend
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Revenue & Net Income Trend**")
            rev_series = extract_series(inc, "total_net_sales")
            ni_series  = extract_series(inc, "net_income")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name  = "Revenue",
                x     = years,
                y     = [v/1000 if v else 0 for v in rev_series],
                marker_color = "#89b4fa",
            ))
            fig.add_trace(go.Bar(
                name  = "Net Income",
                x     = years,
                y     = [v/1000 if v else 0 for v in ni_series],
                marker_color = "#a6e3a1",
            ))
            fig.update_layout(
                barmode    = "group",
                yaxis_title = "USD Billions",
                plot_bgcolor = "rgba(0,0,0,0)",
                paper_bgcolor = "rgba(0,0,0,0)",
                height = 320,
                legend = dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("**Product Segment Revenue**")
            segments = {
                "iPhone":   seg.get("iphone",    {}).get("year_1"),
                "Services": seg.get("services",  {}).get("year_1"),
                "Mac":      seg.get("mac",        {}).get("year_1"),
                "iPad":     seg.get("ipad",       {}).get("year_1"),
                "Wearables":seg.get("wearables_home_acc", {}).get("year_1"),
            }
            seg_clean = {k: v for k, v in segments.items() if v and v > 0}

            if seg_clean:
                fig2 = px.pie(
                    values = list(seg_clean.values()),
                    names  = list(seg_clean.keys()),
                    color_discrete_sequence = px.colors.qualitative.Pastel,
                )
                fig2.update_layout(
                    height = 320,
                    plot_bgcolor  = "rgba(0,0,0,0)",
                    paper_bgcolor = "rgba(0,0,0,0)",
                )
                st.plotly_chart(fig2, use_container_width=True)

        # YoY Changes table
        st.markdown("**Year-over-Year Changes**")
        yoy = financials.get("yoy_changes", {})
        rows = []

        metrics = [
            ("Revenue",      "income_statement",   "total_net_sales"),
            ("Gross Margin", "income_statement",   "gross_margin"),
            ("Net Income",   "income_statement",   "net_income"),
            ("R&D",          "operating_expenses", "research_and_development"),
            ("iPhone",       "product_segments",   "iphone"),
            ("Services",     "product_segments",   "services"),
        ]

        for label, cat, met in metrics:
            pct = yoy.get(cat, {}).get(met, {}).get("yoy_change_pct")
            dirn = yoy.get(cat, {}).get(met, {}).get("direction", "")
            if pct is not None:
                rows.append({"Metric": label, "YoY Change": f"{dirn}{pct:+.1f}%"})

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)


with tab2:
    if not results:
        st.info("Run the pipeline to see sentiment analysis.")
    else:
        from glob import glob
        sent_files = sorted(glob(f"data/processed/{ticker_input}_*_sentiment.json"))

        if sent_files:
            with open(sent_files[-1]) as f:
                sent_full = json.load(f)

            lex = sent_full.get("lexicon", {})
            llm = sent_full.get("llm_analysis", {})
            phr = sent_full.get("phrases", {})

            col_s1, col_s2 = st.columns(2)

            with col_s1:
                st.markdown("**Lexicon Scores (Loughran-McDonald)**")
                lex_data = {
                    "Positive words":      lex.get("positive_count", 0),
                    "Negative words":      lex.get("negative_count", 0),
                    "Uncertainty words":   lex.get("uncertainty_count", 0),
                    "Forward-looking":     lex.get("forward_looking_count", 0),
                    "Litigious words":     lex.get("litigious_count", 0),
                }
                fig3 = px.bar(
                    x      = list(lex_data.values()),
                    y      = list(lex_data.keys()),
                    orientation = "h",
                    color_discrete_sequence = ["#89b4fa"],
                )
                fig3.update_layout(
                    height = 300,
                    plot_bgcolor  = "rgba(0,0,0,0)",
                    paper_bgcolor = "rgba(0,0,0,0)",
                    xaxis_title   = "Word Count",
                    yaxis_title   = "",
                )
                st.plotly_chart(fig3, use_container_width=True)

            with col_s2:
                st.markdown("**LLM Dimensional Analysis**")
                st.metric("Overall Tone",       llm.get("overall_tone", "N/A"))
                st.metric("Tone Score",         f"{llm.get('tone_score', 'N/A')}/10")
                st.metric("Forward Confidence", f"{llm.get('forward_confidence', 'N/A')}/10")
                st.metric("Uncertainty Level",  llm.get("uncertainty_level", "N/A"))
                st.metric("Net Sentiment",      f"{lex.get('net_sentiment_score', 'N/A')}")

            pos_themes = llm.get("key_positive_themes", [])
            concerns   = llm.get("key_concerns", [])

            if pos_themes or concerns:
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    if pos_themes:
                        st.markdown("**✓ Positive Themes**")
                        for t in pos_themes:
                            st.markdown(f"- {t}")
                with col_t2:
                    if concerns:
                        st.markdown("**⚠ Key Concerns**")
                        for c in concerns:
                            st.markdown(f"- {c}")

            notable = llm.get("notable_language", "")
            if notable:
                st.info(f"**Notable language:** {notable}")

        else:
            st.info("No sentiment data available.")


with tab3:
    if report:
        # Fix dollar sign rendering in Streamlit markdown
        report_display = report.replace("$", "\\$")
        st.markdown(report_display)
    else:
        st.info("No report available. Run the pipeline first.")


# ── PDF Export ────────────────────────────────────────────────────────────────
if results and financials and report:
    st.divider()
    st.subheader("📥 Export Report")

    col_dl, col_info = st.columns([1, 3])

    with col_dl:
        if st.button("Generate PDF", type="primary", use_container_width=True):
            with st.spinner("Building PDF report..."):
                try:
                    from src.utils.pdf_generator import generate_pdf
                    from glob import glob
                    import json

                    # Load full sentiment data
                    sent_files = sorted(glob(f"data/processed/{ticker_input}_*_sentiment.json"))
                    full_sentiment = {}
                    if sent_files:
                        with open(sent_files[-1]) as f:
                            full_sentiment = json.load(f)

                    pdf_bytes = generate_pdf(
                        ticker      = ticker_input,
                        company     = financials.get("company", ticker_input),
                        filing_date = financials.get("filing_date", "unknown"),
                        financials  = financials,
                        sentiment   = full_sentiment,
                        report_text = report,
                        confidence  = results.get("financials", {}).get("confidence_score", 1.0),
                        conf_label  = "HIGH",
                    )

                    st.session_state["pdf_bytes"] = pdf_bytes
                    st.success("PDF ready to download!")

                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

    with col_info:
        st.caption("Downloads a formatted PDF including financial tables, sentiment analysis, and the full AI-generated research report.")

    # Show download button once PDF is generated
    if "pdf_bytes" in st.session_state:
        st.download_button(
            label     = "⬇️ Download PDF Report",
            data      = st.session_state["pdf_bytes"],
            file_name = f"AlphaSignal_{ticker_input}_Report.pdf",
            mime      = "application/pdf",
            use_container_width = False,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("AlphaSignal — Multi-agent financial intelligence | Data: SEC EDGAR | LLM: Groq Llama-3.1")