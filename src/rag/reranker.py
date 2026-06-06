"""
Hybrid Re-Ranker for AlphaSignal
-----------------------------------
Standard RAG retrieves by semantic similarity only.
This re-ranker adds a second pass combining:

    Semantic score (ChromaDB cosine similarity)
        Strength: finds conceptually related chunks
        Weakness: misses exact financial terms

    BM25 score (keyword term-frequency matching)
        Strength: exact match on "EPS", "GAAP", "Q3"
        Weakness: misses paraphrased or synonymous text

Hybrid = (0.6 × semantic) + (0.4 × BM25)
Best of both worlds.
"""

import re
import numpy as np
from rank_bm25 import BM25Okapi


# ── Tokenizer ─────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """
    Split text into lowercase word tokens.
    Simple but effective for financial text.
    Example: "Total net sales $391B" → ["total", "net", "sales", "391b"]
    """
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())


# ── BM25 Scoring ──────────────────────────────────────────────────────────────
def compute_bm25_scores(query: str, chunks: list[dict]) -> list[float]:
    """
    Score each chunk against the query using BM25 Okapi algorithm.

    BM25 rewards chunks that:
        - Contain the exact query terms (term frequency)
        - Are not too long (length normalization)
        - Contain rare terms (inverse document frequency)

    This is the same algorithm that powered Google search
    before neural embeddings existed — and it still adds
    significant value alongside semantic search.
    """
    corpus        = [tokenize(c["text"]) for c in chunks]
    bm25          = BM25Okapi(corpus)
    query_tokens  = tokenize(query)
    scores        = bm25.get_scores(query_tokens)
    return scores.tolist()


# ── Normalization ─────────────────────────────────────────────────────────────
def normalize_scores(scores: list[float]) -> list[float]:
    """
    Scale any list of scores to 0–1 using min-max normalization.
    Required before combining semantic and BM25 scores —
    they live on different scales and must be comparable.
    """
    arr           = np.array(scores, dtype=float)
    min_s, max_s  = arr.min(), arr.max()

    if max_s - min_s == 0:
        return [0.5] * len(scores)   # all equal — assign neutral score

    return ((arr - min_s) / (max_s - min_s)).tolist()


# ── Re-Ranker ─────────────────────────────────────────────────────────────────
def rerank(
    query:           str,
    chunks:          list[dict],
    top_k:           int   = 5,
    semantic_weight: float = 0.6,
    bm25_weight:     float = 0.4,
) -> list[dict]:
    """
    Re-rank retrieved chunks using hybrid semantic + BM25 scoring.

    Workflow:
        1. Compute BM25 scores for all chunks
        2. Normalize BM25 and semantic scores to 0-1
        3. Compute weighted hybrid score
        4. Sort by hybrid score and return top_k

    Args:
        query:           the user's question
        chunks:          initial ChromaDB retrieval results
        top_k:           how many chunks to return after re-ranking
        semantic_weight: how much to trust semantic similarity (default 60%)
        bm25_weight:     how much to trust keyword matching   (default 40%)

    Returns:
        Top-k re-ranked chunks, each with hybrid_score and bm25_score added
    """
    if not chunks:
        return []

    if len(chunks) == 1:
        chunks[0]["hybrid_score"] = chunks[0]["relevance"]
        chunks[0]["bm25_score"]   = 0.0
        return chunks

    # Step 1 — BM25 scores
    raw_bm25      = compute_bm25_scores(query, chunks)
    norm_bm25     = normalize_scores(raw_bm25)

    # Step 2 — Semantic scores (already 0-1 from ChromaDB)
    semantic      = [c["relevance"] for c in chunks]
    norm_semantic = normalize_scores(semantic)

    # Step 3 — Hybrid combination
    for i, chunk in enumerate(chunks):
        hybrid                 = (semantic_weight * norm_semantic[i]) + \
                                 (bm25_weight     * norm_bm25[i])
        chunk["bm25_score"]    = round(raw_bm25[i], 4)
        chunk["hybrid_score"]  = round(hybrid, 4)
        chunk["relevance"]     = chunk["hybrid_score"]

    # Step 4 — Sort and return top_k
    reranked = sorted(chunks, key=lambda x: x["hybrid_score"], reverse=True)
    return reranked[:top_k]


# ── Comparison Utility ────────────────────────────────────────────────────────
def compare_rankings(
    query:    str,
    original: list[dict],
    reranked: list[dict],
) -> None:
    """
    Print a side-by-side comparison of original vs re-ranked results.
    Useful for debugging and demonstrating the improvement.
    """
    print(f"\nRe-ranking comparison for: '{query}'")
    print(f"{'─'*60}")
    print(f"{'BEFORE (semantic only)':<35} {'AFTER (hybrid)':<35}")
    print(f"{'─'*60}")

    max_len = max(len(original), len(reranked))
    for i in range(max_len):
        orig = original[i] if i < len(original) else None
        rank = reranked[i] if i < len(reranked) else None

        orig_str = f"{orig['metadata']['section'][:28]:<28} {orig['relevance']:.3f}" if orig else ""
        rank_str = f"{rank['metadata']['section'][:28]:<28} {rank['hybrid_score']:.3f}" if rank else ""

        print(f"{orig_str:<35} {rank_str:<35}")
    print(f"{'─'*60}\n")