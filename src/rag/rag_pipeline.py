"""
RAG Pipeline for AlphaSignal
------------------------------
Connects ChromaDB retrieval to Groq LLM generation.
Every answer includes a confidence score built from:
    - Retrieval score:  how relevant were the retrieved chunks?
    - Grounding score:  how much of the answer came from the context?

This is what separates AlphaSignal from a basic chatbot.
The system knows what it knows — and what it doesn't.
"""

import os
import re
from groq import Groq
from dotenv import load_dotenv
from src.rag.vector_store import query_store
from src.rag.reranker import rerank, compare_rankings

load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────
#GROQ_MODEL        = "llama3-8b-8192"
GROQ_MODEL = "llama-3.1-8b-instant"
TEMPERATURE       = 0.1    # low = factual, precise, no creativity
MAX_TOKENS        = 1000
# N_RETRIEVAL       = 5      # number of chunks to retrieve per question

N_RETRIEVAL    = 10   # retrieve more initially for re-ranking
N_FINAL        = 5    # keep top 5 after re-ranking

# Confidence thresholds
HIGH_CONFIDENCE   = 0.65
LOW_CONFIDENCE    = 0.45   # below this, flag for re-retrieval in Week 3


# ── Groq Client ───────────────────────────────────────────────────────────────
def get_groq_client() -> Groq:
    """Initialize Groq client using API key from .env"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env file.\n"
            "Get a free key at console.groq.com"
        )
    return Groq(api_key=api_key)


# ── Prompt Builder ────────────────────────────────────────────────────────────
def build_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Assemble the system message and user message for the LLM.

    System message: sets the role, rules, and behavior.
    User message:   injects retrieved context + the question.

    The LLM reads both together. By telling it to use ONLY
    the provided context, we prevent hallucination of financial
    figures that aren't in the actual filing.

    Returns: (system_message, user_message)
    """
    system_message = """You are a precise financial analyst assistant for AlphaSignal.

Your job is to answer questions about SEC 10-K filings using ONLY the context provided.

Rules you must follow:
1. Use ONLY information from the provided context. Never use outside knowledge.
2. If the answer is not in the context, say exactly: "I cannot find this in the filing."
3. When citing numbers, state them exactly as they appear in the context.
4. Be concise and precise. Financial professionals value accuracy over elaboration.
5. If multiple relevant figures exist, list all of them.
6. Always mention which section your answer comes from (e.g., Item 7, Item 1A)."""

    # Build context block from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        section  = chunk["metadata"].get("section", "unknown")
        relevance = chunk["relevance"]
        context_parts.append(
            f"[Source {i+1} | Section: {section} | Relevance: {relevance:.2f}]\n"
            f"{chunk['text']}\n"
        )

    context_block = "\n---\n".join(context_parts)

    user_message = f"""CONTEXT FROM SEC 10-K FILING:
{context_block}

QUESTION: {question}

Answer using only the context above. Cite which source number(s) support your answer."""

    return system_message, user_message


# ── Confidence Scoring ────────────────────────────────────────────────────────
def calculate_retrieval_confidence(chunks: list[dict]) -> float:
    """
    Average relevance score across all retrieved chunks.
    Higher = ChromaDB found chunks very similar to the question.
    """
    if not chunks:
        return 0.0
    scores = [c["relevance"] for c in chunks]
    return round(sum(scores) / len(scores), 4)


def calculate_grounding_score(answer: str, chunks: list[dict]) -> float:
    """
    Measure how much of the answer came from the retrieved context.

    Method: Word overlap between the answer and the combined context,
    filtering out common stop words that carry no meaning.

    Score of 1.0 = every meaningful word in the answer exists in context
    Score of 0.0 = the answer shares no words with the context (hallucination risk)
    """
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were',
        'be', 'been', 'has', 'have', 'had', 'it', 'its', 'this', 'that',
        'these', 'those', 'as', 'not', 'no', 'so', 'if', 'than', 'then',
        'i', 'cannot', 'find', 'which', 'also', 'about', 'based', 'their'
    }

    # Meaningful words in the answer
    answer_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', answer.lower()))
    answer_words -= stop_words

    if not answer_words:
        return 0.0

    # All words across retrieved chunks
    context_text  = " ".join([c["text"] for c in chunks]).lower()
    context_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', context_text))

    matched   = answer_words.intersection(context_words)
    grounding = len(matched) / len(answer_words)

    return round(grounding, 4)


def calculate_confidence(
    answer: str,
    chunks: list[dict]
) -> dict:
    """
    Combine retrieval and grounding scores into a final confidence score.

    Weights:
        60% retrieval  — did ChromaDB find relevant chunks?
        40% grounding  — did the LLM actually use those chunks?

    Returns a dict with all scores for full transparency.
    """
    retrieval = calculate_retrieval_confidence(chunks)
    grounding = calculate_grounding_score(answer, chunks)
    combined  = round((retrieval * 0.6) + (grounding * 0.4), 4)

    if combined >= HIGH_CONFIDENCE:
        label = "HIGH"
    elif combined >= LOW_CONFIDENCE:
        label = "MEDIUM"
    else:
        label = "LOW — consider re-retrieval"

    return {
        "retrieval_score": retrieval,
        "grounding_score": grounding,
        "confidence":      combined,
        "label":           label,
    }


# ── Core RAG Query ────────────────────────────────────────────────────────────
def query_rag(
    question: str,
    ticker: str = None,
    n_results: int = N_RETRIEVAL,
    verbose: bool = True
) -> dict:
    """
    Full RAG pipeline: question → retrieve → prompt → generate → score.

    Returns a dict containing:
        answer:      the LLM's response
        confidence:  full confidence breakdown
        sources:     which chunks were retrieved
        question:    the original question
    """
    if verbose:
        print(f"\nQuestion: {question}")
        print("Retrieving relevant chunks...")


    # Step 1: Retrieve broader set
    chunks = query_store(question, n_results=n_results, ticker=ticker)

    if not chunks:
        return {
            "question":   question,
            "answer":     "No relevant context found in the vector store.",
            "confidence": {"confidence": 0.0, "label": "LOW"},
            "sources":    [],
        }

    # Step 1b: Re-rank using hybrid scoring
    original_chunks = [c.copy() for c in chunks]   # save for comparison
    chunks = rerank(question, chunks, top_k=N_FINAL)

    if verbose:
        print(f"Retrieved {len(original_chunks)} chunks → re-ranked to top {len(chunks)}. Querying Groq...")
        compare_rankings(question, original_chunks[:N_FINAL], chunks)
   
   
    # Step 2: Build prompt
    system_message, user_message = build_prompt(question, chunks)

    # Step 3: Generate
    client   = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    answer = response.choices[0].message.content.strip()

    # Step 4: Score confidence
    confidence = calculate_confidence(answer, chunks)

    # Step 5: Package sources for citation
    sources = [
        {
            "section":   c["metadata"].get("section", "unknown"),
            "relevance": c["relevance"],
            "preview":   c["text"][:120] + "...",
        }
        for c in chunks
    ]

    if verbose:
        print(f"Confidence: {confidence['confidence']} ({confidence['label']})")

    return {
        "question":   question,
        "answer":     answer,
        "confidence": confidence,
        "sources":    sources,
    }


# ── Pretty Printer ────────────────────────────────────────────────────────────
def print_result(result: dict) -> None:
    """Print a RAG result in a clean, readable format."""
    conf = result["confidence"]

    print("\n" + "=" * 60)
    print(f"Q: {result['question']}")
    print("=" * 60)
    print(f"\nA: {result['answer']}")
    print("\n" + "-" * 60)
    print(f"CONFIDENCE:  {conf['confidence']}  [{conf['label']}]")
    print(f"  Retrieval: {conf['retrieval_score']}")
    print(f"  Grounding: {conf['grounding_score']}")
    print("\nSOURCES:")
    for i, src in enumerate(result["sources"]):
        print(f"  [{i+1}] {src['section']:<45} relevance: {src['relevance']}")
    print("=" * 60)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AlphaSignal — RAG Pipeline Test")
    print("Connecting ChromaDB retrieval to Groq LLM")
    print("=" * 60)

    # Test questions covering different sections of the 10-K
    test_questions = [
        "What were Apple's total net sales in 2024?",
        "What are Apple's biggest risk factors?",
        "How did iPhone revenue change compared to last year?",
        "What is Apple's strategy for artificial intelligence?",
        "What were the operating expenses?",
    ]

    for question in test_questions:
        result = query_rag(question, ticker="AAPL")
        print_result(result)
        print()