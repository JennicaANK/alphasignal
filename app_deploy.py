"""
AlphaSignal — Hugging Face Spaces Deployment Version
------------------------------------------------------
Self-contained Streamlit app that calls the pipeline
directly (no FastAPI backend required).

On first load: shows pre-computed AAPL results.
Live analysis: runs the full pipeline with GROQ_API_KEY.
"""

import streamlit as st
import json
import os
from glob import glob
from pathlib import Path

st.set_page_config(
    page_title = "AlphaSignal",
    page_icon  = "📈",
    layout     = "wide",
)

st.markdown("""
<style>
.stMetric { background: #1e1e2e; border-radius: 8px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_cached(ticker: str) -> dict:
    """Load pre-computed results from disk."""
    fin_files  = sorted(glob(f"data/processed/{ticker}_*_financials.json"))
    sent_files = sorted(glob(f"data/processed/{ticker}_*_sentiment.json"))
    rep_files  = sorted(glob(f"reports/{ticker}_*_report.md"))

    result = {}

    if fin_files:
        with open(fin_files[-1]) as f:
            result["financials"] = json.load(f)

    if sent_files:
        with open(sent_files[-1]) as f:
            result["sentiment"] = json.load(f)

    if rep_files:
        with open(rep_files[-1]) as f:
            result["report"] = f.read()

    return result


def run_live_pipeline(ticker: str) -> dict:
    """Run the full multi-agent pipeline."""
    from src.agents.graph import build_graph

    pipeline = build_graph()

    final = pipeline.invoke({
        "ticker":           ticker,
        "filing_date":      None,
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
    })
    return load_cached(ticker)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 AlphaSignal")
    st.markdown("*Multi-agent financial intelligence*")
    st.divider()

    has_key = bool(os.getenv("GROQ_API_KEY"))
    if has_key:
        st.success("API key configured")
    else:
        st.warning("No API key — showing cached demo data")

    st.markdown("### Analyze a Company")

    ticker_input = st.text_input(
        "Ticker Symbol",
        value     = "AAPL",
        max_chars = 5,
    ).upper().strip()

    analyze_btn = st.button(
        "Run AlphaSignal Pipeline",
        type    = "primary",
        disabled = not has_key,
        use_container_width = True,
    )

    if not has_key:
        st.caption("Live analysis requires a Groq API key configured as a Space secret.")

    st.divider()
    st.markdown("#### How it works")
    st.markdown("""
1. Downloads 10-K from SEC EDGAR
2. Extracts financial metrics via RAG
3. Analyzes management sentiment
4. Self-checks data quality
5. Generates research report
    """)
    st.divider()
    st.markdown("**Built with:**")
    st.markdown("LangGraph · ChromaDB · Groq · FastAPI · Streamlit")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📈 AlphaSignal")
st.markdown("*Autonomous financial intelligence powered by multi-agent AI*")
st.divider()

# Run pipeline or load cache
if analyze_btn:
    with st.spinner(f"Running AlphaSignal pipeline for {ticker_input}... (2-3 minutes)"):
        try:
            data = run_live_pipeline(ticker_input)
            st.success(f"Analysis complete for {ticker_input}")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            data = load_cached(ticker_input)
else:
    data = load_cached(ticker_input)

if not data:
    st.info(f"No cached data for **{ticker_input}**. Click Run to analyze.")
    st.stop()

financials = data.get("financials", {})
sentiment  = data.get("sentiment",  {})
report     = data.get("report",     "")

inc = financials.get("income_statement", {})
yrs = financials.get("years", {"year_1": "2025", "year_2": "2024", "year_3": "2023"})
sen_lex = sentiment.get("lexicon", {})
sen_llm = sentiment.get("llm_analysis", {})

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.subheader(f"📊 {ticker_input} — Financial Overview")
c1, c2, c3, c4 = st.columns(4)

rev = inc.get("total_net_sales", {}).get("year_1")
ni  = inc.get("net_income",      {}).get("year_1")

c1.metric("Total Revenue",    f"${rev/1000:.1f}B" if rev else "N/A")
c2.metric("Net Income",       f"${ni/1000:.1f}B"  if ni  else "N/A")
c3.metric("Management Tone",  sen_llm.get("overall_tone", "N/A"))
c4.metric("Filing Year",      yrs.get("year_1", "N/A"))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express       as px
import pandas               as pd

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Financial Charts",
    "🧠 Sentiment Analysis",
    "📄 Research Report",
    "📥 Export PDF",
])

with tab1:
    years = [yrs.get("year_3"), yrs.get("year_2"), yrs.get("year_1")]

    def get_series(cat, metric):
        d = financials.get(cat, {}).get(metric, {})
        return [d.get("year_3"), d.get("year_2"), d.get("year_1")]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Revenue & Net Income Trend**")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Revenue",    x=years,
            y=[v/1000 if v else 0 for v in get_series("income_statement","total_net_sales")],
            marker_color="#89b4fa"
        ))
        fig.add_trace(go.Bar(
            name="Net Income", x=years,
            y=[v/1000 if v else 0 for v in get_series("income_statement","net_income")],
            marker_color="#a6e3a1"
        ))
        fig.update_layout(
            barmode="group", yaxis_title="USD Billions",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=300, legend=dict(orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Product Segment Revenue**")
        seg = financials.get("product_segments", {})
        segments = {
            "iPhone":    seg.get("iphone",           {}).get("year_1"),
            "Services":  seg.get("services",         {}).get("year_1"),
            "Mac":       seg.get("mac",              {}).get("year_1"),
            "iPad":      seg.get("ipad",             {}).get("year_1"),
            "Wearables": seg.get("wearables_home_acc",{}).get("year_1"),
        }
        seg_clean = {k: v for k, v in segments.items() if v and v > 0}
        if seg_clean:
            fig2 = px.pie(
                values=list(seg_clean.values()),
                names=list(seg_clean.keys()),
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig2.update_layout(
                height=300,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Year-over-Year Changes**")
    yoy = financials.get("yoy_changes", {})
    rows = []
    for label, cat, met in [
        ("Revenue",      "income_statement",   "total_net_sales"),
        ("Gross Margin", "income_statement",   "gross_margin"),
        ("Net Income",   "income_statement",   "net_income"),
        ("R&D",          "operating_expenses", "research_and_development"),
        ("iPhone",       "product_segments",   "iphone"),
        ("Services",     "product_segments",   "services"),
    ]:
        p = yoy.get(cat, {}).get(met, {}).get("yoy_change_pct")
        d = yoy.get(cat, {}).get(met, {}).get("direction", "")
        if p is not None:
            rows.append({"Metric": label, "YoY Change": f"{d}{p:+.1f}%"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


with tab2:
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Lexicon Scores (Loughran-McDonald)**")
        lex_data = {
            "Positive":     sen_lex.get("positive_count",        0),
            "Negative":     sen_lex.get("negative_count",        0),
            "Uncertainty":  sen_lex.get("uncertainty_count",     0),
            "Fwd-looking":  sen_lex.get("forward_looking_count", 0),
            "Litigious":    sen_lex.get("litigious_count",       0),
        }
        fig3 = px.bar(
            x=list(lex_data.values()), y=list(lex_data.keys()),
            orientation="h", color_discrete_sequence=["#89b4fa"]
        )
        fig3.update_layout(
            height=280, plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Word Count", yaxis_title=""
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_s2:
        st.markdown("**LLM Dimensional Analysis**")
        st.metric("Overall Tone",       sen_llm.get("overall_tone",       "N/A"))
        st.metric("Tone Score",         f"{sen_llm.get('tone_score','N/A')}/10")
        st.metric("Forward Confidence", f"{sen_llm.get('forward_confidence','N/A')}/10")
        st.metric("Net Sentiment",      str(sen_lex.get("net_sentiment_score", "N/A")))

    pos = sen_llm.get("key_positive_themes", [])
    con = sen_llm.get("key_concerns",        [])
    if pos or con:
        c_p, c_c = st.columns(2)
        with c_p:
            if pos:
                st.markdown("**✓ Positive Themes**")
                for t in pos: st.markdown(f"- {t}")
        with c_c:
            if con:
                st.markdown("**⚠ Key Concerns**")
                for c in con: st.markdown(f"- {c}")

    notable = sen_llm.get("notable_language", "")
    if notable:
        st.info(f"**Notable:** {notable}")


with tab3:
    if report:
        report_display = report.replace("$", "\\$")
        st.markdown(report_display)
    else:
        st.info("No report available.")


with tab4:
    st.markdown("### Download Research Report as PDF")
    if st.button("Generate PDF", type="primary"):
        with st.spinner("Building PDF..."):
            try:
                from src.utils.pdf_generator import generate_pdf
                pdf_bytes = generate_pdf(
                    ticker      = ticker_input,
                    company     = financials.get("company", ticker_input),
                    filing_date = financials.get("filing_date", "unknown"),
                    financials  = financials,
                    sentiment   = sentiment,
                    report_text = report,
                    confidence  = 1.0,
                    conf_label  = "HIGH",
                )
                st.session_state["pdf_bytes"] = pdf_bytes
                st.success("PDF ready!")
            except Exception as e:
                st.error(f"PDF error: {e}")

    if "pdf_bytes" in st.session_state:
        st.download_button(
            label     = "⬇️ Download PDF Report",
            data      = st.session_state["pdf_bytes"],
            file_name = f"AlphaSignal_{ticker_input}_Report.pdf",
            mime      = "application/pdf",
        )

st.divider()
st.caption("AlphaSignal — Multi-agent financial intelligence | Data: SEC EDGAR | LLM: Groq Llama-3.1")