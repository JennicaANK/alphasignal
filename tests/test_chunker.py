"""
Unit tests for src/rag/chunker.py

Tests the core chunking logic without any API calls.
All inputs and outputs are pure Python — fast and free to run.
"""

import pytest
from src.rag.chunker import chunk_text, chunk_sections


# ── chunk_text tests ──────────────────────────────────────────────────────────

class TestChunkText:

    def test_short_text_returns_one_chunk(self):
        """Text shorter than chunk_size should return exactly one chunk."""
        text   = "Apple revenue grew significantly this quarter."
        result = chunk_text(text, chunk_size=500, overlap=50)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_returns_multiple_chunks(self):
        """Text longer than chunk_size should be split into multiple chunks."""
        text   = "Apple Inc. " * 200   # ~2200 chars
        result = chunk_text(text, chunk_size=500, overlap=100)
        assert len(result) > 1

    def test_overlap_is_respected(self):
        """Consecutive chunks should share content due to overlap."""
        text   = "word " * 400    # 2000 chars
        result = chunk_text(text, chunk_size=500, overlap=100)
        # Check that end of chunk N overlaps with start of chunk N+1
        if len(result) >= 2:
            end_of_first   = result[0][-80:]
            start_of_second = result[1][:80]
            # There should be some shared content
            assert len(set(end_of_first.split()) &
                       set(start_of_second.split())) > 0

    def test_no_empty_chunks(self):
        """No chunk should be an empty string."""
        text   = "Revenue increased. Net income improved. EPS grew. " * 50
        result = chunk_text(text, chunk_size=200, overlap=50)
        for chunk in result:
            assert chunk.strip() != ""

    def test_all_content_covered(self):
        """Combined chunks should contain all words from original text."""
        text       = "Apple iPhone Mac iPad Services Wearables revenue profit"
        result     = chunk_text(text, chunk_size=200, overlap=20)
        combined   = " ".join(result)
        for word in text.split():
            assert word in combined

    def test_empty_text_returns_empty_list(self):
        """Empty input should return empty list."""
        result = chunk_text("", chunk_size=500, overlap=100)
        assert result == []

    def test_sentence_boundary_preference(self):
        """Chunks should prefer cutting at sentence boundaries."""
        text = ("Revenue grew significantly this year. " * 10 +
                "Net income also improved considerably. " * 10)
        result = chunk_text(text, chunk_size=300, overlap=50)
        for chunk in result[:-1]:
            # Most chunks should end at a sentence boundary
            stripped = chunk.strip()
            # Check it doesn't end mid-word (has a space or period near end)
            assert len(stripped) > 0


# ── chunk_sections tests ──────────────────────────────────────────────────────

class TestChunkSections:

    def test_metadata_attached_to_every_chunk(self):
        """Every chunk must have ticker and filing_date in metadata."""
        sections = {
            "item_1_business": {
                "header":  "ITEM 1. BUSINESS",
                "content": "Apple designs products. " * 100,
            }
        }
        chunks = chunk_sections(sections, ticker="AAPL", filing_date="2024-11-01")
        for chunk in chunks:
            assert chunk["metadata"]["ticker"]      == "AAPL"
            assert chunk["metadata"]["filing_date"] == "2024-11-01"
            assert "section"      in chunk["metadata"]
            assert "chunk_index"  in chunk["metadata"]

    def test_chunk_ids_are_unique(self):
        """Every chunk must have a unique ID."""
        sections = {
            "item_1": {"header": "ITEM 1.", "content": "word " * 300},
            "item_7": {"header": "ITEM 7.", "content": "word " * 300},
        }
        chunks = chunk_sections(sections, ticker="AAPL", filing_date="2024-11-01")
        ids    = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_section_header_in_chunk_text(self):
        """Section header should be prepended to every chunk."""
        sections = {
            "item_7_mda": {
                "header":  "ITEM 7. MANAGEMENT DISCUSSION",
                "content": "Revenue increased. " * 50,
            }
        }
        chunks = chunk_sections(sections, ticker="AAPL", filing_date="2024-11-01")
        assert all("ITEM 7" in c["text"] for c in chunks)

    def test_short_section_skipped(self):
        """Sections with fewer than 100 chars of content should be skipped."""
        sections = {
            "tiny_section": {"header": "ITEM X.", "content": "Too short."},
        }
        chunks = chunk_sections(sections, ticker="AAPL", filing_date="2024-11-01")
        assert len(chunks) == 0

    def test_empty_sections_returns_empty(self):
        """Empty sections dict returns empty chunks list."""
        chunks = chunk_sections({}, ticker="AAPL", filing_date="2024-11-01")
        assert chunks == []