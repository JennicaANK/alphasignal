"""
Sentiment Analyzer for AlphaSignal
-------------------------------------
Analyzes tone and language patterns in SEC 10-K filings,
specifically the MD&A section (Item 7) where management
speaks most directly about results and outlook.

Two complementary approaches:

    Lexicon-based (Loughran-McDonald financial dictionary)
        Fast, interpretable, no API cost
        Counts positive/negative/uncertainty/litigious words
        Designed specifically for financial text — unlike VADER
        or TextBlob which misclassify financial language

    LLM-based dimensional analysis
        Nuanced, catches context word-counting misses
        Scores: optimism, uncertainty, forward confidence
        Identifies key themes and notable language patterns

Why MD&A specifically?
    Item 7 is where executives explain results in their own words.
    Unlike financial tables (which are fixed numbers), MD&A reveals
    whether management is confident, hedging, or downplaying problems.

    A company with declining metrics but confident MD&A language
    signals management believes the decline is temporary.
    Confident metrics with increasingly cautious MD&A is a warning sign.
"""

import re
import json
import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from src.rag.vector_store import query_store
from src.rag.reranker import rerank

load_dotenv()

GROQ_MODEL  = "llama-3.1-8b-instant"
TEMPERATURE = 0.1


# ── Loughran-McDonald Financial Sentiment Lexicon ─────────────────────────────
# These word lists are derived from the Loughran-McDonald (2011) paper
# "When Is a Liability Not a Liability?" — the gold standard for
# financial text sentiment analysis. Unlike general dictionaries,
# these are calibrated for SEC filings specifically.

LM_POSITIVE = {
    "achieve", "achieved", "achieving", "advantage", "beneficial",
    "best", "better", "breakthrough", "confidence", "confident",
    "deliver", "delivered", "delivering", "efficiency", "excellent",
    "exceptional", "favorable", "gain", "gained", "growth",
    "groundbreaking", "improve", "improved", "improving", "increase",
    "increased", "increasing", "innovative", "leader", "leading",
    "momentum", "opportunity", "opportunities", "outstanding",
    "outperform", "outperformed", "positive", "profitable",
    "progress", "profitability", "record", "robust", "significant",
    "solid", "strength", "strong", "stronger", "strongest",
    "succeed", "success", "successful", "superior", "sustainable",
    "transformative", "exceed", "exceeded", "exceeding", "growth",
    "expand", "expanded", "expanding", "expansion"
}

LM_NEGATIVE = {
    "adverse", "against", "challenging", "challenges", "concern",
    "concerning", "decline", "declined", "declining", "decrease",
    "decreased", "decreasing", "deficit", "delay", "delayed",
    "difficult", "difficulties", "difficulty", "disappoint",
    "disappointed", "disappointing", "disruption", "disruptions",
    "downturn", "failure", "harm", "impair", "impaired",
    "impairment", "inadequate", "inflation", "loss", "losses",
    "lower", "negative", "obstacle", "penalty", "poor", "problem",
    "problems", "recession", "restructuring", "risk", "risks",
    "shortage", "slow", "slowdown", "uncertain", "unfavorable",
    "volatile", "volatility", "weakness", "weaken", "weakening",
    "worsen", "worsened", "worsening", "write-off", "writedown"
}

LM_UNCERTAINTY = {
    "approximately", "assume", "assumed", "belief", "believe",
    "believed", "could", "depend", "depends", "estimate",
    "estimated", "estimates", "expect", "expected", "expecting",
    "fluctuate", "fluctuates", "if", "intend", "intends",
    "likely", "may", "might", "ongoing", "plan", "planned",
    "planning", "possible", "potentially", "predict", "projected",
    "projections", "should", "subject", "uncertain", "uncertainty",
    "unclear", "unpredictable", "variable", "whether", "would"
}

LM_LITIGIOUS = {
    "alleged", "allegation", "arbitration", "breach", "claim",
    "claims", "complaint", "court", "damages", "defendant",
    "dispute", "enforcement", "filed", "indemnification",
    "injunction", "judgment", "lawsuit", "legal", "liabilities",
    "liable", "litigation", "penalty", "plaintiff", "proceedings",
    "regulatory", "settlement", "sued", "suit", "tribunal",
    "verdict", "violation"
}

LM_FORWARD_LOOKING = {
    "aim", "anticipate", "anticipated", "anticipating", "believe",
    "commitment", "committed", "continue", "continues", "continuing",
    "deliver", "expect", "expected", "expecting", "focus", "focused",
    "forecast", "future", "goal", "guidance", "intend", "invest",
    "investing", "investment", "launch", "long-term", "objective",
    "opportunity", "outlook", "pipeline", "plan", "planned",
    "position", "priority", "project", "projected", "pursue",
    "roadmap", "strategy", "target", "will", "would"
}


# ── Lexicon Analyzer ──────────────────────────────────────────────────────────
def analyze_lexicon(text: str) -> dict:
    """
    Count financial sentiment words using Loughran-McDonald lexicon.

    Returns counts and ratios for each sentiment category.
    The net sentiment score = (positive - negative) / total_words
    Range: -1.0 (extremely negative) to +1.0 (extremely positive)
    """
    words      = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    total      = len(words)
    word_set   = set(words)

    pos_words  = [w for w in words if w in LM_POSITIVE]
    neg_words  = [w for w in words if w in LM_NEGATIVE]
    unc_words  = [w for w in words if w in LM_UNCERTAINTY]
    lit_words  = [w for w in words if w in LM_LITIGIOUS]
    fwd_words  = [w for w in words if w in LM_FORWARD_LOOKING]

    pos_count  = len(pos_words)
    neg_count  = len(neg_words)

    net_score  = round((pos_count - neg_count) / max(total, 1), 4)
    pos_ratio  = round(pos_count / max(total, 1), 4)
    unc_ratio  = round(len(unc_words) / max(total, 1), 4)

    # Sentiment label based on net score
    if net_score > 0.02:
        label = "POSITIVE"
    elif net_score < -0.02:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "word_count":            total,
        "positive_count":        pos_count,
        "negative_count":        neg_count,
        "uncertainty_count":     len(unc_words),
        "litigious_count":       len(lit_words),
        "forward_looking_count": len(fwd_words),
        "net_sentiment_score":   net_score,
        "positive_ratio":        pos_ratio,
        "uncertainty_ratio":     unc_ratio,
        "sentiment_label":       label,
        "top_positive_words":    list(set(pos_words))[:10],
        "top_negative_words":    list(set(neg_words))[:10],
        "top_uncertainty_words": list(set(unc_words))[:10],
    }


# ── LLM Dimensional Analyzer ──────────────────────────────────────────────────
def analyze_with_llm(text: str, ticker: str) -> dict:
    """
    Use Groq to perform dimensional sentiment analysis on MD&A text.

    Goes beyond word counting to understand:
    - Overall management tone and confidence level
    - Key positive themes being emphasized
    - Concerns being raised or downplayed
    - Whether language is more/less optimistic than typical
    - Notable rhetorical patterns

    Returns structured JSON with dimensional scores and themes.
    """
    system_message = """You are an expert financial analyst specializing in
qualitative analysis of SEC filings. Analyze the provided MD&A text and
return ONLY valid JSON. No preamble, no explanation, no markdown.

Return exactly this JSON structure:
{
  "overall_tone": "BULLISH|CAUTIOUSLY_OPTIMISTIC|NEUTRAL|CAUTIOUS|BEARISH",
  "tone_score": <1-10 where 10 is most optimistic>,
  "forward_confidence": <1-10 where 10 is most confident about future>,
  "uncertainty_level": "LOW|MEDIUM|HIGH",
  "key_positive_themes": [<3 themes management emphasizes positively>],
  "key_concerns": [<3 areas of concern or risk mentioned>],
  "notable_language": "<one sentence about the most striking language pattern>",
  "vs_typical_disclosure": "MORE_OPTIMISTIC|TYPICAL|MORE_CAUTIOUS",
  "management_credibility_signals": "<brief assessment of hedging vs directness>"
}"""

    user_message = f"""Analyze this {ticker} 10-K MD&A section for tone and sentiment:

{text[:3500]}

Return ONLY valid JSON. No other text."""

    try:
        client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=600,
        )
        raw = response.choices[0].message.content

        # Strip markdown if present
        raw = re.sub(r'^```json\s*', '', raw.strip())
        raw = re.sub(r'^```\s*',     '', raw)
        raw = re.sub(r'\s*```$',     '', raw)

        return json.loads(raw.strip())

    except Exception as e:
        print(f"  LLM analysis failed: {e}")
        return {
            "overall_tone":       "UNKNOWN",
            "tone_score":         None,
            "forward_confidence": None,
            "error":              str(e)
        }


# ── Context Retriever ─────────────────────────────────────────────────────────
def get_mda_text(ticker: str) -> str:
    """
    Retrieve MD&A chunks from the vector store.
    Item 7 is the most linguistically rich section of any 10-K.
    """
    queries = [
        "management discussion analysis results operations",
        "revenue growth performance outlook future",
        "challenges risks macro environment headwinds tailwinds",
    ]

    seen   = set()
    chunks = []

    for query in queries:
        raw      = query_store(query, n_results=4, ticker=ticker)
        reranked = rerank(query, raw, top_k=2)
        for c in reranked:
            uid = f"{c['metadata'].get('section')}_{c['metadata'].get('chunk_index')}"
            if uid not in seen:
                seen.add(uid)
                chunks.append(c["text"])

    return "\n\n".join(chunks)


# ── Phrase Extractor ──────────────────────────────────────────────────────────
def extract_sentiment_phrases(text: str, n: int = 5) -> dict:
    """
    Extract the most sentiment-rich sentences from the text.
    Returns top positive and negative sentences for the report.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)

    pos_sentences = []
    neg_sentences = []

    for sent in sentences:
        if len(sent) < 30 or len(sent) > 300:
            continue
        words    = set(re.findall(r'\b[a-zA-Z]+\b', sent.lower()))
        pos_hits = len(words & LM_POSITIVE)
        neg_hits = len(words & LM_NEGATIVE)

        if pos_hits > neg_hits and pos_hits >= 2:
            pos_sentences.append((pos_hits, sent.strip()))
        elif neg_hits > pos_hits and neg_hits >= 2:
            neg_sentences.append((neg_hits, sent.strip()))

    pos_sentences.sort(reverse=True)
    neg_sentences.sort(reverse=True)

    return {
        "top_positive_sentences": [s for _, s in pos_sentences[:n]],
        "top_negative_sentences": [s for _, s in neg_sentences[:n]],
    }


# ── Main Analyzer ─────────────────────────────────────────────────────────────
def analyze_sentiment(
    ticker:      str,
    filing_date: str = None,
    output_dir:  str = "data/processed"
) -> dict:
    """
    Full sentiment analysis pipeline.
    1. Retrieve MD&A text via RAG
    2. Lexicon-based scoring
    3. LLM dimensional analysis
    4. Extract notable phrases
    5. Combine and save
    """
    print(f"\n{'='*55}")
    print(f"AlphaSignal — Sentiment Analyzer: {ticker}")
    print(f"{'='*55}")

    # Step 1: Get MD&A text
    print("\nStep 1: Retrieving MD&A section...")
    text = get_mda_text(ticker)
    print(f"  Retrieved {len(text):,} characters of management language")

    if not text:
        print("  No MD&A text found.")
        return {}

    # Step 2: Lexicon analysis
    print("Step 2: Running Loughran-McDonald lexicon analysis...")
    lexicon = analyze_lexicon(text)
    print(f"  Positive words:    {lexicon['positive_count']}")
    print(f"  Negative words:    {lexicon['negative_count']}")
    print(f"  Uncertainty words: {lexicon['uncertainty_count']}")
    print(f"  Net score:         {lexicon['net_sentiment_score']} ({lexicon['sentiment_label']})")

    # Step 3: LLM analysis
    print("Step 3: Running LLM dimensional analysis...")
    llm_result = analyze_with_llm(text, ticker)
    print(f"  Overall tone:      {llm_result.get('overall_tone', 'N/A')}")
    print(f"  Tone score:        {llm_result.get('tone_score', 'N/A')}/10")
    print(f"  Fwd confidence:    {llm_result.get('forward_confidence', 'N/A')}/10")

    # Step 4: Extract notable phrases
    print("Step 4: Extracting sentiment-rich phrases...")
    phrases = extract_sentiment_phrases(text)
    print(f"  Found {len(phrases['top_positive_sentences'])} positive sentences")
    print(f"  Found {len(phrases['top_negative_sentences'])} negative sentences")

    # Step 5: Combine results
    result = {
        "ticker":        ticker,
        "filing_date":   filing_date or "unknown",
        "section":       "MD&A (Item 7)",
        "text_length":   len(text),
        "lexicon":       lexicon,
        "llm_analysis":  llm_result,
        "phrases":       phrases,
    }

    # Step 6: Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    date_str   = filing_date or "latest"
    out_path   = os.path.join(output_dir, f"{ticker}_{date_str}_sentiment.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


# ── Pretty Printer ────────────────────────────────────────────────────────────
def print_sentiment(result: dict) -> None:
    """Print sentiment analysis in a clean readable format."""
    if not result:
        return

    lex = result.get("lexicon", {})
    llm = result.get("llm_analysis", {})
    phr = result.get("phrases", {})

    print(f"\n{'='*60}")
    print(f"  SENTIMENT ANALYSIS — {result['ticker']} MD&A")
    print(f"{'='*60}")

    print(f"\n  LEXICON SCORES  (Loughran-McDonald)")
    print(f"  {'─'*40}")
    print(f"  Net sentiment score:   {lex.get('net_sentiment_score'):>8}  [{lex.get('sentiment_label')}]")
    print(f"  Positive words:        {lex.get('positive_count'):>8}")
    print(f"  Negative words:        {lex.get('negative_count'):>8}")
    print(f"  Uncertainty words:     {lex.get('uncertainty_count'):>8}")
    print(f"  Litigious words:       {lex.get('litigious_count'):>8}")
    print(f"  Forward-looking words: {lex.get('forward_looking_count'):>8}")
    print(f"  Uncertainty ratio:     {lex.get('uncertainty_ratio'):>8}")

    print(f"\n  LLM DIMENSIONAL ANALYSIS")
    print(f"  {'─'*40}")
    print(f"  Overall tone:          {llm.get('overall_tone', 'N/A')}")
    print(f"  Tone score:            {llm.get('tone_score', 'N/A')}/10")
    print(f"  Forward confidence:    {llm.get('forward_confidence', 'N/A')}/10")
    print(f"  Uncertainty level:     {llm.get('uncertainty_level', 'N/A')}")
    print(f"  vs Typical:            {llm.get('vs_typical_disclosure', 'N/A')}")

    themes = llm.get("key_positive_themes", [])
    if themes:
        print(f"\n  KEY POSITIVE THEMES")
        for t in themes:
            print(f"    ✓  {t}")

    concerns = llm.get("key_concerns", [])
    if concerns:
        print(f"\n  KEY CONCERNS")
        for c in concerns:
            print(f"    ⚠  {c}")

    notable = llm.get("notable_language", "")
    if notable:
        print(f"\n  NOTABLE LANGUAGE")
        print(f"    {notable}")

    credibility = llm.get("management_credibility_signals", "")
    if credibility:
        print(f"\n  MANAGEMENT SIGNALS")
        print(f"    {credibility}")

    pos_phrases = phr.get("top_positive_sentences", [])
    if pos_phrases:
        print(f"\n  TOP POSITIVE PHRASES")
        for p in pos_phrases[:3]:
            print(f"    + {p[:120]}")

    neg_phrases = phr.get("top_negative_sentences", [])
    if neg_phrases:
        print(f"\n  TOP NEGATIVE PHRASES")
        for n in neg_phrases[:3]:
            print(f"    - {n[:120]}")

    print(f"\n{'='*60}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = analyze_sentiment(ticker="AAPL", filing_date="2024-11-01")
    print_sentiment(result)