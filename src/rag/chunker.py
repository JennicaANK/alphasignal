"""
Text Chunker for AlphaSignal RAG Pipeline
------------------------------------------
Splits cleaned 10-K sections into overlapping chunks
optimized for vector embedding and semantic retrieval.

Why overlap?
    If a key financial figure sits at a chunk boundary,
    overlap ensures it appears in both adjacent chunks
    so it is never lost during retrieval.

Chunk size choice:
    500 words (~2000 chars) balances two things:
    - Small enough: embeddings capture focused meaning
    - Large enough: each chunk has useful context
"""

import json
import os
import re
from pathlib import Path


# ── Chunking Parameters ───────────────────────────────────────────────────────
CHUNK_SIZE    = 1500   # characters per chunk (~350 words)
CHUNK_OVERLAP = 300    # characters of overlap between adjacent chunks


# ── Core Chunking Logic ───────────────────────────────────────────────────────
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Split a long string into overlapping chunks.

    Algorithm:
    1. Start at position 0
    2. Take chunk_size characters
    3. Try to cut at a sentence boundary (. ? !)
       instead of mid-word for cleaner chunks
    4. Move forward by (chunk_size - overlap) characters
    5. Repeat until the end of the text
    """
    chunks = []
    start  = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Find the last sentence boundary before the cut point
            boundary = max(
                text.rfind('. ', start, end),
                text.rfind('? ', start, end),
                text.rfind('! ', start, end),
            )
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + 1   # include the period

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Slide forward, keeping the overlap
        start = end - overlap
        if start >= len(text):
            break

    return chunks


def chunk_sections(
    sections: dict,
    ticker: str,
    filing_date: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """
    Chunk every section and attach metadata to each chunk.

    Metadata is critical — it tells the RAG system WHERE
    a chunk came from so results can be cited and traced.

    Each chunk becomes a dict:
    {
        "id":       "AAPL_2024-11-01_item_1_business_chunk_3",
        "text":     "Apple designs, manufactures and markets...",
        "metadata": {
            "ticker":       "AAPL",
            "filing_date":  "2024-11-01",
            "section":      "item_1_business",
            "chunk_index":  3,
            "total_chunks": 12,
            "char_count":   1487
        }
    }
    """
    all_chunks = []

    for section_key, section_data in sections.items():
        content = section_data.get("content", "")
        header  = section_data.get("header",  "")

        if not content or len(content) < 100:
            continue

        # Prepend the section header to every chunk
        # so the LLM always knows which section it's reading
        text_with_header = f"{header}\n\n{content}"

        raw_chunks = chunk_text(text_with_header, chunk_size, overlap)
        total      = len(raw_chunks)

        for i, chunk_text_val in enumerate(raw_chunks):
            chunk_id = f"{ticker}_{filing_date}_{section_key}_chunk_{i}"

            all_chunks.append({
                "id":   chunk_id,
                "text": chunk_text_val,
                "metadata": {
                    "ticker":        ticker,
                    "filing_date":   filing_date,
                    "section":       section_key,
                    "section_header": header,
                    "chunk_index":   i,
                    "total_chunks":  total,
                    "char_count":    len(chunk_text_val),
                }
            })

    return all_chunks


def load_sections(sections_path: str) -> dict:
    """Load the sections JSON produced by text_parser.py (Day 3)."""
    with open(sections_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_chunks(chunks: list[dict], output_dir: str = "data/processed") -> str:
    """Save all chunks to a JSON file for inspection and debugging."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not chunks:
        print("No chunks to save.")
        return ""

    ticker       = chunks[0]["metadata"]["ticker"]
    filing_date  = chunks[0]["metadata"]["filing_date"]
    output_path  = os.path.join(output_dir, f"{ticker}_{filing_date}_chunks.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return output_path


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import glob

    print("=" * 55)
    print("AlphaSignal — Text Chunker")
    print("=" * 55)

    # Find the most recent sections file from Day 3
    section_files = sorted(glob.glob("data/processed/*_sections.json"))

    if not section_files:
        print("No sections file found. Run text_parser.py first.")
    else:
        sections_path = section_files[-1]
        print(f"Loading sections from: {sections_path}")

        # Extract ticker and date from filename
        filename   = Path(sections_path).stem
        parts      = filename.split("_")
        ticker     = parts[0]
        filing_date = parts[1] if len(parts) > 1 else "unknown"

        sections = load_sections(sections_path)
        chunks   = chunk_sections(sections, ticker, filing_date)

        print(f"\nChunking results:")
        print(f"  Sections processed:  {len(sections)}")
        print(f"  Total chunks created: {len(chunks)}")
        print(f"  Avg chunk size:       {sum(c['metadata']['char_count'] for c in chunks) // max(len(chunks),1)} chars")

        # Show a sample chunk
        if chunks:
            sample = chunks[2]
            print(f"\nSample chunk (index 2):")
            print(f"  ID:      {sample['id']}")
            print(f"  Section: {sample['metadata']['section']}")
            print(f"  Size:    {sample['metadata']['char_count']} chars")
            print(f"  Preview: {sample['text'][:200]}...")

        output_path = save_chunks(chunks)
        print(f"\nAll chunks saved to: {output_path}")

        print("\n" + "=" * 55)
        print(f"DONE — {len(chunks)} chunks ready for ChromaDB")
        print("=" * 55)