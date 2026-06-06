"""
Text Parser for SEC 10-K Filings
---------------------------------------
Reads raw HTML filings downloaded from SEC EDGAR,
strips HTML tags, cleans noise, and extracts the
key sections financial analysts actually care about.

10-K filings follow a strict SEC structure:
    Item 1   — Business Overview
    Item 1A  — Risk Factors
    Item 7   — Management Discussion & Analysis (MD&A)
    Item 7A  — Market Risk
    Item 8   — Financial Statements
"""

import re
import os
import json
import glob
from pathlib import Path
from bs4 import BeautifulSoup


# ── Section targets ───────────────────────────────────────────────────────────
# These are the parts of a 10-K most valuable for financial intelligence.
# Item 7 (MD&A) is where management explains results in plain English.
# Item 1A is where they disclose every risk they face.
TARGET_SECTIONS = [
    "item 1",
    "item 1a",
    "item 7",
    "item 7a",
    "item 8",
]


# ── Step 1: Read ──────────────────────────────────────────────────────────────
def read_raw_filing(file_path: str) -> str:
    """Read raw filing text from disk."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ── Step 2: Strip HTML ────────────────────────────────────────────────────────
def strip_html(raw_text: str) -> str:
    """
    Remove all HTML tags and extract readable plain text.
    Uses BeautifulSoup with the lxml parser (fast and lenient).
    Also removes script/style blocks which contain no useful content.
    """
    soup = BeautifulSoup(raw_text, "lxml")

    # Remove tags that never contain readable content
    for tag in soup(["script", "style", "meta", "link", "ix:header"]):
        tag.decompose()

    # get_text() pulls all visible text, separator adds newlines between blocks
    text = soup.get_text(separator="\n")
    return text


# ── Step 3: Clean ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove noise lines.

    10-K filings are full of:
    - Page numbers (just a digit on a line)
    - Repeated dashes and underscores (table borders)
    - Hundreds of consecutive blank lines
    - Lines with only punctuation

    We strip all of that here so the RAG system
    only sees meaningful financial language.
    """
    lines = [line.strip() for line in text.splitlines()]

    cleaned = []
    for line in lines:
        # Skip empty lines
        if not line:
            continue
        # Skip lines that are only digits, dashes, dots (page numbers/borders)
        if re.match(r'^[\d\s\.\-\_\=\*\/\\]+$', line):
            continue
        # Skip very short noise lines (single chars, etc.)
        if len(line) < 3:
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)

    # Collapse 3+ blank lines into a single blank line
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces into one
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


# ── Step 4: Extract Sections ──────────────────────────────────────────────────
def extract_sections(text: str) -> dict:
    """
    Split the cleaned 10-K text into its standard SEC sections.

    Every 10-K filed with the SEC must follow the same structure.
    We find each 'ITEM X.' header and grab all text until the next header.
    This gives us clean, labeled chunks for the RAG pipeline.

    Example output:
    {
        "item_1_business": {
            "header": "ITEM 1. BUSINESS",
            "content": "Apple designs, manufactures...",
            "char_count": 24500
        },
        ...
    }
    """
    # Match patterns like: ITEM 1., ITEM 1A., Item 7., ITEM 7A.
    item_pattern = re.compile(
        r'(ITEM\s+\d+[A-Z]?[\.\:]\s*.{3,60}?)(?:\n|$)',
        re.IGNORECASE
    )

    matches = list(item_pattern.finditer(text))

    if not matches:
        print("  Warning: No standard Item sections found. Saving full text.")
        return {"full_text": {"header": "Full Document", "content": text[:15000], "char_count": len(text)}}

    sections = {}
    for i, match in enumerate(matches):
        header = match.group(0).strip()
        start  = match.end()
        end    = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        content = text[start:end].strip()

        # Only keep sections with real content (skip table of contents entries)
        if len(content) < 150:
            continue

        # Build a clean dictionary key from the header
        key = re.sub(r'[^a-z0-9]+', '_', header[:40].lower()).strip('_')
        sections[key] = {
            "header":     header,
            "content":    content[:12000],  # cap per section — enough for RAG
            "char_count": len(content),
        }

    return sections


# ── Step 5: Save Outputs ──────────────────────────────────────────────────────
def save_outputs(
    text: str,
    sections: dict,
    base_name: str,
    output_dir: str = "data/processed"
) -> dict:
    """Save clean text and sections JSON to data/processed/."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    clean_path    = os.path.join(output_dir, f"{base_name}_clean.txt")
    sections_path = os.path.join(output_dir, f"{base_name}_sections.json")

    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(text)

    with open(sections_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    return {"clean_text_path": clean_path, "sections_path": sections_path}


# ── Full Pipeline ─────────────────────────────────────────────────────────────
def parse_filing(raw_file_path: str, output_dir: str = "data/processed") -> dict:
    """
    Master function — runs all 5 steps in sequence.
    Input:  path to raw .txt filing in data/raw/
    Output: clean text + sections JSON in data/processed/
    """
    print(f"\nParsing: {raw_file_path}")

    # 1. Read
    raw_text = read_raw_filing(raw_file_path)
    print(f"  Raw size:      {len(raw_text):>10,} characters")

    # 2. Strip HTML
    is_html = "<html" in raw_text[:2000].lower()
    if is_html:
        print("  Format:        HTML — stripping tags...")
        text = strip_html(raw_text)
    else:
        print("  Format:        Plain text")
        text = raw_text

    # 3. Clean
    text = clean_text(text)
    print(f"  After clean:   {len(text):>10,} characters")
    reduction = round((1 - len(text) / len(raw_text)) * 100, 1)
    print(f"  Noise removed: {reduction}%")

    # 4. Extract sections
    sections = extract_sections(text)
    print(f"  Sections found: {len(sections)}")
    for key, val in sections.items():
        print(f"    {key[:45]:<45} {val['char_count']:>8,} chars")

    # 5. Save
    base_name = Path(raw_file_path).stem
    paths = save_outputs(text, sections, base_name, output_dir)

    print(f"\n  Saved clean text: {paths['clean_text_path']}")
    print(f"  Saved sections:   {paths['sections_path']}")

    return {
        "raw_file":        raw_file_path,
        "clean_text_path": paths["clean_text_path"],
        "sections_path":   paths["sections_path"],
        "raw_chars":       len(raw_text),
        "clean_chars":     len(text),
        "noise_removed":   f"{reduction}%",
        "sections_found":  len(sections),
    }


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw_files = sorted(glob.glob("data/raw/*.txt"))

    print("=" * 55)
    print("AlphaSignal — Text Parser")
    print("=" * 55)

    if not raw_files:
        print("No raw filings found in data/raw/")
        print("Run src/utils/sec_fetcher.py first.")
    else:
        # Parse the most recently downloaded filing
        raw_file = raw_files[-1]
        result = parse_filing(raw_file)

        print("\n" + "=" * 55)
        print("DONE")
        print(f"  Raw:            {result['raw_chars']:,} chars")
        print(f"  Clean:          {result['clean_chars']:,} chars")
        print(f"  Noise removed:  {result['noise_removed']}")
        print(f"  Sections:       {result['sections_found']}")
        print("=" * 55)