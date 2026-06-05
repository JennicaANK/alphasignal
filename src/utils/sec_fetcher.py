"""
SEC EDGAR API Fetcher
Fetches real 10-K annual filings directly from the SEC's free public API.
No API key required. No cost.
"""

import requests
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# SEC requires a User-Agent header identifying your app and email
USER_AGENT = os.getenv("SEC_USER_AGENT", "AlphaSignal contact@example.com")

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}


def get_company_cik(ticker: str) -> str:
    """
    Convert a stock ticker to SEC CIK number.
    CIK is the unique company ID used by SEC EDGAR.
    Example: AAPL -> 0000320193
    """
    print(f"Looking up CIK for ticker: {ticker.upper()}")

    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    data = response.json()
    ticker_upper = ticker.upper()

    for entry in data.values():
        if entry["ticker"] == ticker_upper:
            # CIK must be zero-padded to 10 digits for API calls
            cik = str(entry["cik_str"]).zfill(10)
            print(f"Found CIK: {cik} for {entry['title']}")
            return cik

    raise ValueError(f"Ticker '{ticker}' not found in SEC database.")


def get_latest_10k_metadata(cik: str) -> dict:
    """
    Get metadata for the most recent 10-K filing.
    Returns accession number, filing date, primary document filename.
    """
    print(f"Fetching filing history for CIK: {cik}")

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    data = response.json()
    filings = data["filings"]["recent"]

    forms            = filings["form"]
    accession_nums   = filings["accessionNumber"]
    filing_dates     = filings["filingDate"]
    primary_docs     = filings["primaryDocument"]

    # Find the most recent 10-K
    for i, form in enumerate(forms):
        if form == "10-K":
            metadata = {
                "company_name":     data["name"],
                "cik":              cik,
                "accession_number": accession_nums[i],
                "filing_date":      filing_dates[i],
                "primary_document": primary_docs[i],
            }
            print(f"Found 10-K: {metadata['company_name']} filed {metadata['filing_date']}")
            return metadata

    raise ValueError(f"No 10-K filing found for CIK {cik}")


def download_10k_text(ticker: str, output_dir: str = "data/raw") -> dict:
    """
    Full pipeline: ticker -> CIK -> filing metadata -> download text.
    Saves raw text to data/raw/ (gitignored).
    Returns metadata dict with file path included.
    """
    # Step 1: Get CIK
    cik = get_company_cik(ticker)
    time.sleep(0.5)  # Be polite to SEC servers

    # Step 2: Get latest 10-K metadata
    metadata = get_latest_10k_metadata(cik)
    time.sleep(0.5)

    # Step 3: Build the document URL
    acc_no_clean = metadata["accession_number"].replace("-", "")
    doc_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{acc_no_clean}/{metadata['primary_document']}"
    )

    print(f"Downloading filing from: {doc_url}")
    response = requests.get(doc_url, headers=HEADERS)
    response.raise_for_status()

    # Step 4: Save raw text to disk
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{ticker.upper()}_{metadata['filing_date']}_10K.txt"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(response.text)

    size_kb = len(response.text) / 1024
    print(f"Saved to: {output_path}")
    print(f"File size: {size_kb:,.1f} KB")

    metadata["file_path"] = output_path
    metadata["file_size_kb"] = round(size_kb, 1)
    return metadata


def save_metadata(metadata: dict, output_dir: str = "data/processed") -> str:
    """
    Save filing metadata as JSON to data/processed/.
    This DOES go to GitHub (unlike raw filings).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ticker = metadata["company_name"].replace(" ", "_")
    filename = f"{ticker}_{metadata['filing_date']}_metadata.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Test with Apple — change ticker to any company you like
    TEST_TICKER = "AAPL"

    print("=" * 50)
    print(f"AlphaSignal — SEC EDGAR Fetcher Test")
    print(f"Fetching 10-K for: {TEST_TICKER}")
    print("=" * 50)

    metadata = download_10k_text(TEST_TICKER)
    save_metadata(metadata)

    print("\n" + "=" * 50)
    print("SUCCESS")
    print(f"Company:      {metadata['company_name']}")
    print(f"Filed:        {metadata['filing_date']}")
    print(f"File size:    {metadata['file_size_kb']} KB")
    print(f"Raw file:     {metadata['file_path']}")
    print("=" * 50)