"""
PDF Report Generator for AlphaSignal
---------------------------------------
Generates a professionally formatted PDF research report
from the structured pipeline output.

Uses reportlab — already in requirements.txt.
Builds the PDF directly from structured data rather than
converting markdown, which gives precise layout control.

Output: a PDF bytes object that Streamlit can offer as a download.
"""

from reportlab.lib.pagesizes   import letter
from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units       import inch
from reportlab.lib             import colors
from reportlab.platypus        import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.enums       import TA_CENTER, TA_LEFT, TA_RIGHT
from io                        import BytesIO
from datetime                  import datetime


# ── Color palette ─────────────────────────────────────────────────────────────
DARK_BG     = colors.HexColor("#1e1e2e")
ACCENT_BLUE = colors.HexColor("#89b4fa")
ACCENT_GREEN= colors.HexColor("#a6e3a1")
ACCENT_RED  = colors.HexColor("#f38ba8")
TEXT_DARK   = colors.HexColor("#11111b")
TEXT_GRAY   = colors.HexColor("#6c7086")
WHITE       = colors.white
LIGHT_GRAY  = colors.HexColor("#f5f5f5")
BORDER_GRAY = colors.HexColor("#cdd6f4")


def build_styles():
    """Define all paragraph styles used in the report."""
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "title",
            parent    = base["Title"],
            fontSize  = 24,
            textColor = TEXT_DARK,
            spaceAfter = 6,
            fontName  = "Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent    = base["Normal"],
            fontSize  = 11,
            textColor = TEXT_GRAY,
            spaceAfter = 4,
            fontName  = "Helvetica",
        ),
        "section": ParagraphStyle(
            "section",
            parent    = base["Heading2"],
            fontSize  = 13,
            textColor = TEXT_DARK,
            spaceBefore = 14,
            spaceAfter  = 6,
            fontName  = "Helvetica-Bold",
            borderPad = 4,
        ),
        "body": ParagraphStyle(
            "body",
            parent    = base["Normal"],
            fontSize  = 9.5,
            textColor = TEXT_DARK,
            spaceAfter = 4,
            leading   = 14,
            fontName  = "Helvetica",
        ),
        "bold": ParagraphStyle(
            "bold",
            parent    = base["Normal"],
            fontSize  = 9.5,
            textColor = TEXT_DARK,
            fontName  = "Helvetica-Bold",
        ),
        "small": ParagraphStyle(
            "small",
            parent    = base["Normal"],
            fontSize  = 8,
            textColor = TEXT_GRAY,
            fontName  = "Helvetica",
        ),
        "highlight": ParagraphStyle(
            "highlight",
            parent    = base["Normal"],
            fontSize  = 9.5,
            textColor = TEXT_DARK,
            backColor = LIGHT_GRAY,
            borderPad = 6,
            fontName  = "Helvetica-Oblique",
            spaceAfter = 8,
        ),
    }
    return styles


def make_table(data, col_widths=None, header=True):
    """Build a styled reportlab table."""
    table = Table(data, colWidths=col_widths)

    style = [
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("GRID",        (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]

    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT_BLUE),
            ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, 0), 9.5),
        ]

    table.setStyle(TableStyle(style))
    return table


def generate_pdf(
    ticker:     str,
    company:    str,
    filing_date: str,
    financials: dict,
    sentiment:  dict,
    report_text: str,
    confidence: float = 1.0,
    conf_label: str   = "HIGH",
) -> bytes:
    """
    Generate a complete PDF research report.

    Args:
        ticker:      stock ticker (e.g. "AAPL")
        company:     company name (e.g. "Apple Inc.")
        filing_date: SEC filing date
        financials:  structured financial data dict
        sentiment:   sentiment analysis results dict
        report_text: the generated markdown report text
        confidence:  pipeline confidence score
        conf_label:  confidence label (HIGH/MEDIUM/LOW)

    Returns:
        PDF as bytes — pass directly to st.download_button
    """
    buffer = BytesIO()
    doc    = SimpleDocTemplate(
        buffer,
        pagesize     = letter,
        rightMargin  = 0.75 * inch,
        leftMargin   = 0.75 * inch,
        topMargin    = 0.75 * inch,
        bottomMargin = 0.75 * inch,
    )

    styles  = build_styles()
    story   = []
    w       = 7.0 * inch   # usable width

    # ── Cover / Header ────────────────────────────────────────────────────────
    story.append(Paragraph("AlphaSignal", styles["title"]))
    story.append(Paragraph(
        "Multi-Agent Financial Intelligence System",
        styles["subtitle"]
    ))
    story.append(HRFlowable(width=w, thickness=1.5, color=ACCENT_BLUE))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"<b>Research Report: {company} ({ticker})</b>",
        styles["section"]
    ))

    # Metadata table
    meta = [
        ["Filing Date", filing_date,
         "Generated",   datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Pipeline Confidence", f"{confidence:.4f} [{conf_label}]",
         "Data Source", "SEC EDGAR 10-K"],
        ["LLM", "Groq Llama-3.1-8b-instant",
         "System", "AlphaSignal v1.0"],
    ]
    story.append(make_table(
        meta,
        col_widths = [1.4*inch, 2.1*inch, 1.4*inch, 2.1*inch],
        header     = False,
    ))
    story.append(Spacer(1, 14))

    # ── Financial Performance ─────────────────────────────────────────────────
    story.append(HRFlowable(width=w, thickness=0.5, color=BORDER_GRAY))
    story.append(Paragraph("Financial Performance", styles["section"]))

    inc  = financials.get("income_statement",   {})
    opex = financials.get("operating_expenses", {})
    yrs  = financials.get("years", {"year_1": "2025", "year_2": "2024", "year_3": "2023"})
    yoy  = financials.get("yoy_changes",        {})
    y1, y2, y3 = yrs.get("year_1","Y1"), yrs.get("year_2","Y2"), yrs.get("year_3","Y3")

    def fmt(category, metric, year="year_1"):
        try:
            v = financials.get(category, {}).get(metric, {}).get(year)
            return f"${v:,.0f}M" if isinstance(v, (int, float)) else "N/A"
        except Exception:
            return "N/A"

    def fmt_pct(category, metric):
        try:
            p = yoy.get(category, {}).get(metric, {}).get("yoy_change_pct")
            d = yoy.get(category, {}).get(metric, {}).get("direction", "")
            return f"{d}{p:+.1f}%" if p is not None else "N/A"
        except Exception:
            return "N/A"

    fin_data = [
        ["Metric", y1, y2, y3, "YoY Change"],
        ["Total Revenue",
         fmt("income_statement","total_net_sales"),
         fmt("income_statement","total_net_sales","year_2"),
         fmt("income_statement","total_net_sales","year_3"),
         fmt_pct("income_statement","total_net_sales")],
        ["Gross Margin",
         fmt("income_statement","gross_margin"),
         fmt("income_statement","gross_margin","year_2"),
         fmt("income_statement","gross_margin","year_3"),
         fmt_pct("income_statement","gross_margin")],
        ["Net Income",
         fmt("income_statement","net_income"),
         fmt("income_statement","net_income","year_2"),
         fmt("income_statement","net_income","year_3"),
         fmt_pct("income_statement","net_income")],
        ["Operating Income",
         fmt("income_statement","operating_income"),
         fmt("income_statement","operating_income","year_2"),
         fmt("income_statement","operating_income","year_3"),
         fmt_pct("income_statement","operating_income")],
        ["R&D Expense",
         fmt("operating_expenses","research_and_development"),
         fmt("operating_expenses","research_and_development","year_2"),
         fmt("operating_expenses","research_and_development","year_3"),
         fmt_pct("operating_expenses","research_and_development")],
        ["SG&A",
         fmt("operating_expenses","selling_general_admin"),
         fmt("operating_expenses","selling_general_admin","year_2"),
         fmt("operating_expenses","selling_general_admin","year_3"),
         fmt_pct("operating_expenses","selling_general_admin")],
    ]

    story.append(make_table(
        fin_data,
        col_widths = [1.8*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.0*inch],
    ))

    # ── Product Segments ──────────────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(Paragraph("Product Segment Revenue", styles["section"]))

    seg = financials.get("product_segments", {})
    seg_data = [
        ["Segment", y1, y2, "YoY Change"],
        ["iPhone",
         fmt("product_segments","iphone"),
         fmt("product_segments","iphone","year_2"),
         fmt_pct("product_segments","iphone")],
        ["Services",
         fmt("product_segments","services"),
         fmt("product_segments","services","year_2"),
         fmt_pct("product_segments","services")],
        ["Mac",
         fmt("product_segments","mac"),
         fmt("product_segments","mac","year_2"),
         fmt_pct("product_segments","mac")],
        ["iPad",
         fmt("product_segments","ipad"),
         fmt("product_segments","ipad","year_2"),
         fmt_pct("product_segments","ipad")],
        ["Wearables & Home",
         fmt("product_segments","wearables_home_acc"),
         fmt("product_segments","wearables_home_acc","year_2"),
         fmt_pct("product_segments","wearables_home_acc")],
    ]

    story.append(make_table(
        seg_data,
        col_widths = [2.0*inch, 1.5*inch, 1.5*inch, 1.5*inch],
    ))

    # ── Sentiment Analysis ────────────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width=w, thickness=0.5, color=BORDER_GRAY))
    story.append(Paragraph("Management Tone & Sentiment Analysis", styles["section"]))

    lex = sentiment.get("lexicon", {})
    llm = sentiment.get("llm_analysis", {})

    sent_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Overall Tone",   llm.get("overall_tone","N/A"),
         "Tone Score",     f"{llm.get('tone_score','N/A')}/10"],
        ["Fwd Confidence", f"{llm.get('forward_confidence','N/A')}/10",
         "Uncertainty",    llm.get("uncertainty_level","N/A")],
        ["Net Sentiment",  str(lex.get("net_sentiment_score","N/A")),
         "Sentiment Label",lex.get("sentiment_label","N/A")],
        ["Positive Words", str(lex.get("positive_count",0)),
         "Negative Words", str(lex.get("negative_count",0))],
        ["Uncertainty Words", str(lex.get("uncertainty_count",0)),
         "Fwd-Looking Words", str(lex.get("forward_looking_count",0))],
    ]

    story.append(make_table(
        sent_data,
        col_widths = [1.6*inch, 1.9*inch, 1.6*inch, 1.9*inch],
    ))

    # Positive themes and concerns
    pos_themes = llm.get("key_positive_themes", [])
    concerns   = llm.get("key_concerns",        [])

    if pos_themes or concerns:
        story.append(Spacer(1, 8))
        tc_data = [["✓ Positive Themes", "⚠ Key Concerns"]]
        max_rows = max(len(pos_themes), len(concerns))
        for i in range(max_rows):
            tc_data.append([
                pos_themes[i] if i < len(pos_themes) else "",
                concerns[i]   if i < len(concerns)   else "",
            ])
        story.append(make_table(
            tc_data,
            col_widths = [3.5*inch, 3.5*inch],
        ))

    notable = llm.get("notable_language", "")
    if notable:
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<i>Notable language: {notable}</i>", styles["highlight"]))

    # ── Research Report ───────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("AI-Generated Research Report", styles["section"]))
    story.append(HRFlowable(width=w, thickness=0.5, color=BORDER_GRAY))
    story.append(Spacer(1, 8))

    # Strip markdown formatting for PDF text rendering
    import re
    clean_lines = []
    for line in report_text.splitlines():
        line = line.strip()
        if not line or line.startswith("---"):
            if clean_lines:
                clean_lines.append("")
            continue
        # Remove markdown headers (##, #)
        line = re.sub(r'^#{1,3}\s+', '', line)
        # Remove bold/italic markers
        line = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', line)
        # Remove markdown table rows
        if line.startswith("|"):
            continue
        # Remove HTML tags
        line = re.sub(r'<[^>]+>', '', line)
        if line:
            clean_lines.append(line)

    for line in clean_lines:
        if line:
            story.append(Paragraph(line, styles["body"]))
        else:
            story.append(Spacer(1, 6))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width=w, thickness=0.5, color=BORDER_GRAY))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Report generated by AlphaSignal on {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Data: SEC EDGAR | LLM: Groq Llama-3.1-8b-instant | "
        f"Pipeline Confidence: {confidence:.4f} [{conf_label}]",
        styles["small"]
    ))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()