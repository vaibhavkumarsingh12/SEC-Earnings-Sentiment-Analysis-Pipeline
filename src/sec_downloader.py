"""
SEC Filing Downloader
=====================
Downloads 10-K and 20-F filings from SEC EDGAR for each company
defined in the project configuration using ``sec-edgar-downloader``.

Usage:
    python -m src.sec_downloader          # from project root
    python src/sec_downloader.py          # direct execution
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Config import – works whether invoked as a module or as a standalone script.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

from sec_edgar_downloader import Downloader  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ---------------------------------------------------------------------------
# Constants from config (re-exported for readability)
# ---------------------------------------------------------------------------
SEC_IDENTITY_NAME: str = config.SEC_IDENTITY_NAME
SEC_IDENTITY_EMAIL: str = config.SEC_IDENTITY_EMAIL
COMPANIES: dict = config.COMPANIES
FILINGS_DIR: Path = config.FILINGS_DIR
NUM_FILINGS: int = config.NUM_FILINGS
SEC_REQUEST_DELAY: float = config.SEC_REQUEST_DELAY


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def download_all_filings(tickers: List[str] | None = None) -> Dict[str, List[Path]]:
    """Download SEC filings for every company listed in ``config.COMPANIES``.

    Creates a :class:`sec_edgar_downloader.Downloader` using the configured
    SEC identity and iterates over each company, downloading the specified
    filing type (``10-K`` or ``20-F``).

    Parameters
    ----------
    tickers : list[str], optional
        If provided, only download filings for these tickers.

    Returns
    -------
    dict[str, list[Path]]
        Mapping of ticker symbols to lists of downloaded file paths.
    """
    dl = Downloader(SEC_IDENTITY_NAME, SEC_IDENTITY_EMAIL, str(FILINGS_DIR))
    results: Dict[str, List[Path]] = {}

    target_companies = {t: COMPANIES[t] for t in tickers if t in COMPANIES} if tickers else COMPANIES
    for ticker, company_info in target_companies.items():
        filing_type: str = company_info["filing_type"]
        logger.info(
            "Downloading %s filings for %s (%s) …",
            filing_type,
            ticker,
            company_info["name"],
        )

        try:
            dl.get(filing_type, ticker, limit=NUM_FILINGS)
            # Collect whatever was saved to disk
            paths = [fp for _, fp in find_filing_paths(ticker)]
            results[ticker] = paths
            logger.info(
                "  ✓ %s — found %d filing document(s) on disk.",
                ticker,
                len(paths),
            )
        except Exception:
            logger.exception("  ✗ Failed to download filings for %s.", ticker)
            results[ticker] = []

        # Respect SEC rate limits
        time.sleep(SEC_REQUEST_DELAY)

    return results


def find_filing_paths(ticker: str) -> List[Tuple[int, Path]]:
    """Locate downloaded filing documents for *ticker* on disk.

    The ``sec-edgar-downloader`` library (v5+) stores filings in a nested
    directory structure under ``FILINGS_DIR``::

        FILINGS_DIR/
        └─ sec-edgar-filings/
           └─ <TICKER>/
              └─ <FILING_TYPE>/
                 └─ <ACCESSION_NO>/
                    ├─ filing-details.html   ← skip this
                    ├─ primary-document.html  ← keep
                    └─ …

    The function walks the tree and returns all ``.txt``, ``.htm``, and
    ``.html`` files (excluding ``filing-details.html``), sorted by the
    year embedded in the accession-number directory name.

    Parameters
    ----------
    ticker : str
        Company ticker symbol (e.g. ``"JPM"``).

    Returns
    -------
    list[tuple[int, Path]]
        Sorted list of ``(year, filepath)`` tuples.  If the year cannot be
        determined from the path, ``0`` is used as a fallback.
    """
    filing_type = COMPANIES.get(ticker, {}).get("filing_type", "10-K")
    base_dir = FILINGS_DIR / "sec-edgar-filings" / ticker / filing_type

    if not base_dir.exists():
        logger.warning("Directory not found for %s: %s", ticker, base_dir)
        return []

    valid_extensions = {".txt", ".htm", ".html"}
    skip_filenames = {"filing-details.html"}
    found: List[Tuple[int, Path]] = []

    for filepath in base_dir.rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in valid_extensions:
            continue
        if filepath.name.lower() in skip_filenames:
            continue

        # Try to extract a year from the accession-number directory name
        # Accession numbers look like  0000019617-24-000047
        year = _extract_year_from_path(filepath)
        found.append((year, filepath))

    found.sort(key=lambda t: t[0])
    return found


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

_YEAR_RE = re.compile(r"(?:^|-)(\d{2})(?:-\d+)?$")


def _extract_year_from_path(filepath: Path) -> int:
    """Heuristically extract a filing year from the directory structure.

    The ``sec-edgar-downloader`` accession-number directories contain a
    two-digit year component (e.g. ``0000019617-24-000047`` → 2024).
    As a fallback, any 4-digit year (2000–2099) found in the path is used.

    Returns ``0`` when no year can be determined.
    """
    for part in filepath.parts:
        match = _YEAR_RE.search(part)
        if match:
            short_year = int(match.group(1))
            return 2000 + short_year if short_year < 100 else short_year

    # Fallback: look for a full 4-digit year anywhere in the path string
    four_digit = re.search(r"(20\d{2})", str(filepath))
    if four_digit:
        return int(four_digit.group(1))

    return 0


# ═══════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Starting SEC filing download for %d companies …", len(COMPANIES))
    results = download_all_filings()

    total = sum(len(v) for v in results.values())
    logger.info(
        "Download complete — %d document(s) across %d ticker(s).",
        total,
        len(results),
    )
    for ticker, paths in results.items():
        logger.info("  %s: %d file(s)", ticker, len(paths))
