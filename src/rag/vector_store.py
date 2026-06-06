"""
Vector Store for AlphaSignal RAG Pipeline
-------------------------------------------
Embeds text chunks using sentence-transformers
and stores them in ChromaDB for semantic retrieval.

How it works:
    Text chunk → Embedding model → Vector [0.23, -0.81, 0.44, ...]
    Question   → Embedding model → Vector [0.21, -0.79, 0.41, ...]
    ChromaDB finds chunks whose vectors are closest to the question vector.
    Those chunks are the most semantically relevant passages.
"""

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import json
import glob


# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH      = "chroma_db"          # local folder (gitignored)
COLLECTION_NAME  = "sec_filings"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"  # small, fast, free, runs locally


# ── Embedding Function ────────────────────────────────────────────────────────
def get_embedding_function():
    """
    Load the sentence-transformer embedding model.
    'all-MiniLM-L6-v2' converts any text into a 384-dimensional vector.
    It runs entirely on your local machine — no API call, no cost.
    First run downloads the model (~80MB). Cached after that.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


# ── Client & Collection ───────────────────────────────────────────────────────
def get_collection(reset: bool = False):
    """
    Create or connect to a ChromaDB collection.
    PersistentClient saves vectors to disk so they
    survive between runs — you don't re-embed every time.

    reset=True wipes the collection (use when re-ingesting a filing).
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = get_embedding_function()

    if reset:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"Existing collection '{COLLECTION_NAME}' cleared.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for text
    )
    return collection


# ── Add Documents ─────────────────────────────────────────────────────────────
def add_chunks_to_store(chunks: list[dict], reset: bool = False) -> None:
    """
    Embed all chunks and store them in ChromaDB.

    ChromaDB.add() takes:
        documents: the raw text (gets embedded automatically)
        metadatas: dicts of extra info (ticker, section, date)
        ids:       unique string ID per chunk (no duplicates allowed)
    """
    collection = get_collection(reset=reset)

    # Check how many documents are already in the collection
    existing = collection.count()
    print(f"Collection currently holds: {existing} chunks")

    documents = [c["text"]     for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids       = [c["id"]       for c in chunks]

    # Add in batches of 100 to avoid memory issues
    batch_size = 100
    total      = len(chunks)

    print(f"Embedding and storing {total} chunks...")
    print(f"(First run downloads ~80MB model — this takes 1-2 minutes)\n")

    for i in range(0, total, batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids  = ids[i:i + batch_size]

        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )

        done = min(i + batch_size, total)
        print(f"  Stored {done}/{total} chunks...")

    print(f"\nDone. Collection now holds: {collection.count()} chunks")


# ── Query ─────────────────────────────────────────────────────────────────────
def query_store(question: str, n_results: int = 5, ticker: str = None) -> list[dict]:
    """
    Convert a question to a vector and retrieve the
    most semantically similar chunks from ChromaDB.

    Returns a list of result dicts, each containing:
        text, metadata, and distance score (lower = more similar)
    """
    collection = get_collection()

    # Optional: filter by ticker so we only search one company's filings
    where = {"ticker": ticker} if ticker else None

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        where=where,
    )

    # Reformat results into clean list of dicts
    output = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        output.append({
            "text":      doc,
            "metadata":  meta,
            "distance":  round(dist, 4),
            "relevance": round(1 - dist, 4),   # 1.0 = perfect match
        })

    return output


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # from src.rag.chunker import load_sections, chunk_sections, load_sections

    print("=" * 55)
    print("AlphaSignal — Vector Store Builder")
    print("=" * 55)

    # Load chunks from Day 4 chunker output
    chunk_files = sorted(glob.glob("data/processed/*_chunks.json"))

    if not chunk_files:
        print("No chunks file found. Run chunker.py first.")
    else:
        chunks_path = chunk_files[-1]
        print(f"Loading chunks from: {chunks_path}")

        with open(chunks_path, "r") as f:
            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks")

        # Embed and store (reset=True to rebuild cleanly)
        add_chunks_to_store(chunks, reset=True)

        # ── Test Retrieval ────────────────────────────────────────────────────
        print("\n" + "─" * 55)
        print("Testing retrieval with sample questions...")
        print("─" * 55)

        test_questions = [
            "What were the total revenues?",
            "What are the main risk factors?",
            "How did iPhone sales perform?",
        ]

        for question in test_questions:
            print(f"\nQ: {question}")
            results = query_store(question, n_results=2)
            for i, r in enumerate(results):
                print(f"  Result {i+1} | section: {r['metadata']['section']} | relevance: {r['relevance']}")
                print(f"  Preview: {r['text'][:150]}...")

        print("\n" + "=" * 55)
        print("SUCCESS — Vector store is live and retrieving correctly")
        print("=" * 55)