"""
Central Configuration for SEC Earnings Sentiment Analysis Pipeline
===================================================================
All tunable parameters, file paths, and company metadata live here.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# SEC EDGAR Identity (Required — 403 without this)
# ─────────────────────────────────────────────────────────────────────
SEC_IDENTITY_NAME = "vaibhav kumar singh"
SEC_IDENTITY_EMAIL = "vaibhavkumarsingh235@gmail.com"
SEC_USER_AGENT = f"{SEC_IDENTITY_NAME} {SEC_IDENTITY_EMAIL}"

# ─────────────────────────────────────────────────────────────────────
# Project Root & Directory Layout
# ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
FILINGS_DIR = DATA_DIR / "filings"
MDA_TEXTS_DIR = DATA_DIR / "mda_texts"
SENTIMENT_DIR = DATA_DIR / "sentiment_scores"
MARKET_DATA_DIR = DATA_DIR / "market_data"

OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create all directories on import
for _dir in [FILINGS_DIR, MDA_TEXTS_DIR, SENTIMENT_DIR,
             MARKET_DATA_DIR, PLOTS_DIR, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Company Metadata
# ─────────────────────────────────────────────────────────────────────
COMPANIES = {
    "JPM": {
        "name": "JPMorgan Chase & Co.",
        "cik": "0000019617",
        "filing_type": "10-K",
        "benchmark": "XLF",
    },
    "GS": {
        "name": "Goldman Sachs Group Inc.",
        "cik": "0000886982",
        "filing_type": "10-K",
        "benchmark": "XLF",
    },
    "WFC": {
        "name": "Wells Fargo & Co.",
        "cik": "0000072971",
        "filing_type": "10-K",
        "benchmark": "XLF",
    },
    "C": {
        "name": "Citigroup Inc.",
        "cik": "0000831001",
        "filing_type": "10-K",
        "benchmark": "XLF",
    },
    "HSBC": {
        "name": "HSBC Holdings plc",
        "cik": "0001070205",
        "filing_type": "20-F",
        "benchmark": "XLF",
    },
}

# ─────────────────────────────────────────────────────────────────────
# Date Range
# ─────────────────────────────────────────────────────────────────────
FILING_START_YEAR = 2022
FILING_END_YEAR = 2026  # Inclusive — filings for calendar submission years 2022 through 2026
NUM_FILINGS = 5  # Maximum filings to download per company

# ─────────────────────────────────────────────────────────────────────
# SEC Rate Limiting
# ─────────────────────────────────────────────────────────────────────
SEC_REQUEST_DELAY = 0.15  # Seconds between requests (SEC allows 10/s)

# ─────────────────────────────────────────────────────────────────────
# FinBERT Model Configuration
# ─────────────────────────────────────────────────────────────────────
FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_MAX_TOKENS = 400        # Conservative limit (model max is 512)
FINBERT_OVERLAP_SENTENCES = 1   # Sentences of overlap between chunks
FINBERT_BATCH_SIZE = 8          # Reduce if running on CPU / low VRAM

# ─────────────────────────────────────────────────────────────────────
# Loughran-McDonald Dictionary
# ─────────────────────────────────────────────────────────────────────
LM_DICTIONARY_URL = (
    "https://drive.google.com/uc?export=download&id=12ECPJMxV2wSalXG8ykMmkpa1fq_ur0Rf"
)
LM_DICTIONARY_PATH = DATA_DIR / "LoughranMcDonald_MasterDictionary_2020.csv"

# ─────────────────────────────────────────────────────────────────────
# Market Data Parameters
# ─────────────────────────────────────────────────────────────────────
CAR_WINDOWS = [(-1, 3), (-1, 5), (-1, 10), (-1, 30), (0, 60)]
SECTOR_BENCHMARK = "XLF"  # Financial Select Sector SPDR Fund

# ─────────────────────────────────────────────────────────────────────
# MD&A Extraction Parameters
# ─────────────────────────────────────────────────────────────────────
MDA_MIN_LENGTH = 5000    # Characters — below this, likely a TOC hit
MDA_MAX_LENGTH = 500000  # Characters — above this, likely got too much

# ─────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────
PLOT_DPI = 300
PLOT_STYLE = "darkgrid"
PLOT_PALETTE = "muted"
FIGURE_SIZE = (10, 7)
