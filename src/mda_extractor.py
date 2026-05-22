"""
MD&A Section Extractor
======================
Extracts, cleans, and persists the *Management's Discussion & Analysis*
(MD&A) section from SEC 10-K and 20-F filings downloaded by
:mod:`sec_downloader`.

For 10-K filings the MD&A lives under **Item 7**; for 20-F filings
(e.g. HSBC) it is **Item 5 — Operating and Financial Review**.

Usage:
    python -m src.mda_extractor           # from project root
    python src/mda_extractor.py           # direct execution
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config import
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ---------------------------------------------------------------------------
# Constants from config
# ---------------------------------------------------------------------------
MDA_TEXTS_DIR: Path = config.MDA_TEXTS_DIR
MDA_MIN_LENGTH: int = config.MDA_MIN_LENGTH
MDA_MAX_LENGTH: int = config.MDA_MAX_LENGTH
COMPANIES: dict = config.COMPANIES

# ---------------------------------------------------------------------------
# Regex patterns for section boundaries
# ---------------------------------------------------------------------------
# 10-K  —  Item 7 → Item 8
_10K_START_RE = re.compile(
    r"item\s*7[\s\.\-\–\—\:]*management",
    re.IGNORECASE,
)
_10K_END_RE = re.compile(
    r"item\s*8[\s\.\-\–\—\:]*financial\s+statements",
    re.IGNORECASE,
)

# 20-F  —  Item 5 → Item 6
_20F_START_RE = re.compile(
    r"item\s*5[\s\.\-\–\—\:]*operating\s+and\s+financial",
    re.IGNORECASE,
)
_20F_END_RE = re.compile(
    r"item\s*6[\s\.\-\–\—\:]*directors",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def extract_mda(filepath: Path, filing_type: str = "10-K") -> Optional[str]:
    """Extract the MD&A section from a downloaded SEC filing.

    Uses a fast multi-strategy parser:
    Strategy 1: Standard Item-to-Item Search (Item 7 -> Item 8, or Item 5 -> Item 6)
    Strategy 2: MD&A Incorporated by Reference / Header-less Title Search
    Strategy 3: Fallback (Relaxed Criteria)
    """
    try:
        raw_content = Path(filepath).read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.error("Cannot read file: %s", filepath)
        return None

    # --- Pre-extract main document from SGML wrapper for 100x speedup ---
    main_doc = _extract_main_document(raw_content, filing_type)

    # --- Parse HTML → plain text ---
    text = _html_to_text(main_doc)

    # --- Pre-define start and end patterns ---
    if filing_type.upper() == "20-F":
        # HSBC Operating and Financial Review (Item 5) -> Directors (Item 6)
        start_patterns = [
            re.compile(r"\bItem\s*5[\s\.\-\–\—\:]*Operating\s+and\s+Financial", re.IGNORECASE),
            re.compile(r"\bItem\s*5[\s\.\-\–\—\:]*Operating\s+Review", re.IGNORECASE),
            re.compile(r"\bFinancial\s+review\b", re.IGNORECASE),
        ]
        end_patterns = [
            re.compile(r"\bItem\s*6[\s\.\-\–\—\:]*Directors", re.IGNORECASE),
            re.compile(r"\bItem\s*6[\s\.\-\–\—\:]*Board\s+of\s+Directors", re.IGNORECASE),
            re.compile(r"\bReport\s+of\s+the\s+Directors\b", re.IGNORECASE),
            re.compile(r"\bRisk\s+review\b", re.IGNORECASE),
        ]
    else:
        # 10-K Item 7 -> Item 8
        start_patterns = [
            re.compile(r"\bItem\s*7[\s\.\-\–\—\:]*Management", re.IGNORECASE),
        ]
        end_patterns = [
            re.compile(r"\bItem\s*8[\s\.\-\–\—\:]*Financial\s+Statements", re.IGNORECASE),
            re.compile(r"Report\s+of\s+Independent\s+Registered\s+Public\s+Accounting\s+Firm", re.IGNORECASE),
            re.compile(r"\bCONSOLIDATED\s+FINANCIAL\s+STATEMENTS\b", re.IGNORECASE),
        ]

    # --- STRATEGY 1: Standard Item-to-Item Search ---
    best_candidate = None
    best_len = 0
    best_strategy = None
    
    for start_re in start_patterns:
        start_matches = list(start_re.finditer(text))
        for end_re in end_patterns:
            end_matches = list(end_re.finditer(text))
            
            for sm in start_matches:
                nearest_end = None
                for em in end_matches:
                    if em.start() > sm.start():
                        nearest_end = em
                        break
                if nearest_end:
                    candidate = text[sm.start():nearest_end.start()].strip()
                    candidate_len = len(candidate)
                    if 10000 <= candidate_len <= 500000:
                        if candidate_len > best_len:
                            best_len = candidate_len
                            best_candidate = candidate
                            best_strategy = f"Strategy 1 ({start_re.pattern} -> {end_re.pattern})"

    if best_candidate:
        logger.info("  ✓ MD&A extracted successfully using %s.", best_strategy)
        return best_candidate

    # --- STRATEGY 2: MD&A Incorporated by Reference / Header-less ---
    mda_title_re = re.compile(r"\bManagement['’]s\s+Discussion\s+and\s+Analysis\b", re.IGNORECASE)
    title_matches = list(mda_title_re.finditer(text))
    
    for sm in title_matches:
        for end_re in end_patterns:
            end_matches = list(end_re.finditer(text))
            nearest_end = None
            for em in end_matches:
                if em.start() > sm.start():
                    nearest_end = em
                    break
            if nearest_end:
                candidate = text[sm.start():nearest_end.start()].strip()
                candidate_len = len(candidate)
                if 10000 <= candidate_len <= 500000:
                    if candidate_len > best_len:
                        best_len = candidate_len
                        best_candidate = candidate
                        best_strategy = f"Strategy 2 (Management Title -> {end_re.pattern})"

    if best_candidate:
        logger.info("  ✓ MD&A extracted successfully using %s.", best_strategy)
        return best_candidate

    # --- STRATEGY 3: Fallback (Relaxed Criteria) ---
    for start_re in start_patterns:
        start_matches = list(start_re.finditer(text))
        for end_re in end_patterns:
            end_matches = list(end_re.finditer(text))
            for sm in start_matches:
                nearest_end = None
                for em in end_matches:
                    if em.start() > sm.start():
                        nearest_end = em
                        break
                if nearest_end:
                    candidate = text[sm.start():nearest_end.start()].strip()
                    candidate_len = len(candidate)
                    if 2000 <= candidate_len <= 500000:
                        if candidate_len > best_len:
                            best_len = candidate_len
                            best_candidate = candidate
                            best_strategy = f"Strategy 3 ({start_re.pattern} -> {end_re.pattern})"

    if best_candidate:
        logger.info("  ✓ MD&A extracted successfully using %s.", best_strategy)
        return best_candidate

    logger.warning("No valid MD&A span found in %s", filepath)
    return None


def clean_mda_text(raw_text: str) -> str:
    """Clean extracted MD&A text for downstream NLP processing.

    Steps:
    1. Strip residual HTML tags.
    2. Remove lines that look like tabular / financial data.
    3. Remove page headers, footers, and standalone page numbers.
    4. Collapse excessive whitespace.

    Parameters
    ----------
    raw_text : str
        The raw MD&A text produced by :func:`extract_mda`.

    Returns
    -------
    str
        Cleaned, analysis-ready text.
    """
    # 1. Remove residual HTML tags
    text = _strip_html_tags(raw_text)

    # 2. Line-level filtering
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Skip lines that are mostly numbers / table data
        if _is_table_line(stripped):
            continue

        # Skip page headers / footers
        if _is_header_footer(stripped):
            continue

        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # 3. Collapse multiple blank lines and spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def process_all_filings(tickers: list[str] | None = None) -> Dict[Tuple[str, int], Path]:
    """Extract and clean MD&A from every downloaded filing.

    Iterates over :pydata:`config.COMPANIES`, locates filings via
    :func:`sec_downloader.find_filing_paths`, and writes cleaned text
    to :pydata:`config.MDA_TEXTS_DIR`.

    Parameters
    ----------
    tickers : list[str], optional
        If provided, only process these tickers.

    Returns
    -------
    dict[tuple[str, int], Path]
        Mapping of ``(ticker, year)`` → saved text file path.
    """
    from src.sec_downloader import find_filing_paths  # noqa: E402  (lazy import to avoid circular deps)

    saved: Dict[Tuple[str, int], Path] = {}

    target_companies = {t: COMPANIES[t] for t in tickers if t in COMPANIES} if tickers else COMPANIES
    for ticker, company_info in target_companies.items():
        filing_type: str = company_info["filing_type"]
        filing_paths = find_filing_paths(ticker)

        if not filing_paths:
            logger.warning("No filings found on disk for %s.", ticker)
            continue

        logger.info(
            "Processing %d filing(s) for %s (%s) …",
            len(filing_paths),
            ticker,
            company_info["name"],
        )

        for year, fpath in filing_paths:
            logger.info("  Extracting MD&A from %s (year=%d) …", fpath.name, year)

            raw_mda = extract_mda(fpath, filing_type=filing_type)
            if raw_mda is None:
                logger.warning("  ✗ Extraction failed for %s year %d.", ticker, year)
                continue

            cleaned = clean_mda_text(raw_mda)
            if not cleaned:
                logger.warning(
                    "  ✗ Cleaned text is empty for %s year %d.", ticker, year
                )
                continue

            # Truncate excessively long extractions
            if len(cleaned) > MDA_MAX_LENGTH:
                logger.warning(
                    "  ⚠ Text for %s year %d is %d chars — truncating to %d.",
                    ticker,
                    year,
                    len(cleaned),
                    MDA_MAX_LENGTH,
                )
                cleaned = cleaned[: MDA_MAX_LENGTH]

            out_path = MDA_TEXTS_DIR / f"{ticker}_{year}.txt"
            out_path.write_text(cleaned, encoding="utf-8")
            saved[(ticker, year)] = out_path

            logger.info(
                "  ✓ Saved %s_%d.txt — %d chars.",
                ticker,
                year,
                len(cleaned),
            )

    return saved


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════


def _extract_main_document(raw_content: str, filing_type: str) -> str:
    """Extract the main document block of type *filing_type* from SGML submission."""
    doc_start_indices = []
    start_pos = 0
    while True:
        pos = raw_content.find("<DOCUMENT>", start_pos)
        if pos == -1:
            pos = raw_content.find("<document>", start_pos)
        if pos == -1:
            break
        doc_start_indices.append(pos)
        start_pos = pos + 10

    if not doc_start_indices:
        return raw_content

    for start_idx in doc_start_indices:
        end_idx = raw_content.find("</DOCUMENT>", start_idx)
        if end_idx == -1:
            end_idx = raw_content.find("</document>", start_idx)
        if end_idx == -1:
            end_idx = len(raw_content)
        
        block = raw_content[start_idx:end_idx]
        type_pos = block.find("<TYPE>")
        if type_pos == -1:
            type_pos = block.find("<type>")
        
        if type_pos != -1:
            type_str = block[type_pos+6:type_pos+20].strip().upper()
            if type_str.startswith(filing_type.upper()):
                text_pos = block.find("<TEXT>")
                if text_pos == -1:
                    text_pos = block.find("<text>")
                
                if text_pos != -1:
                    text_end = block.find("</TEXT>", text_pos)
                    if text_end == -1:
                        text_end = block.find("</text>", text_pos)
                    if text_end != -1:
                        return block[text_pos+6:text_end].strip()
                    else:
                        return block[text_pos+6:].strip()
                return block.strip()
                
    return raw_content


def _html_to_text(html: str) -> str:
    """Parse *html* and return plain text with newline separators, with a 25MB truncation safeguard."""
    if len(html) > 25000000:
        logger.info("  ⚠ HTML document size exceeds 25MB (%d chars) — truncating for parser safety", len(html))
        html = html[:25000000]
    try:
        from bs4 import BeautifulSoup  # noqa: E402

        soup = BeautifulSoup(html, "lxml")
    except Exception:
        from bs4 import BeautifulSoup  # noqa: E402, F811

        soup = BeautifulSoup(html, "html.parser")
    return soup.get_text("\n")


def _strip_html_tags(text: str) -> str:
    """Remove any stray HTML tags from *text*."""
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup(text, "html.parser").get_text()
    except Exception:
        return re.sub(r"<[^>]+>", "", text)


# Heuristics for table / financial lines
_DOLLAR_OR_PERCENT_RE = re.compile(r"[\$%]")
_MOSTLY_NUMBERS_RE = re.compile(r"^[\d\s\t\$%\.,\(\)\-–—]+$")


def _is_table_line(line: str) -> bool:
    """Return ``True`` if *line* looks like tabular financial data."""
    # Lines with 3+ occurrences of $ or %
    if len(_DOLLAR_OR_PERCENT_RE.findall(line)) >= 3:
        return True
    # Lines that are almost entirely numbers / punctuation
    if _MOSTLY_NUMBERS_RE.match(line) and len(line) > 5:
        return True
    return False


_HEADER_FOOTER_RE = re.compile(
    r"^(table\s+of\s+contents|\d{1,3}\s*$|page\s+\d+)",
    re.IGNORECASE,
)


def _is_header_footer(line: str) -> bool:
    """Return ``True`` if *line* is a page header/footer artefact."""
    return bool(_HEADER_FOOTER_RE.match(line))


# ═══════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Starting MD&A extraction for %d companies …", len(COMPANIES))
    saved_files = process_all_filings()

    logger.info(
        "Extraction complete — %d MD&A text(s) saved to %s",
        len(saved_files),
        MDA_TEXTS_DIR,
    )
    for (ticker, year), fpath in sorted(saved_files.items()):
        logger.info("  %s %d → %s", ticker, year, fpath)
