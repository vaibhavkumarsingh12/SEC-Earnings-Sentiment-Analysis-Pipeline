"""
Market Data Module — SEC Earnings Sentiment Analysis Pipeline
==============================================================
Fetches stock prices via yfinance, computes Cumulative Abnormal Returns
(CAR) around filing dates, and extracts EPS surprise data.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

from config import (
    COMPANIES,
    MARKET_DATA_DIR,
    CAR_WINDOWS,
    SECTOR_BENCHMARK,
    FILING_START_YEAR,
    FILING_END_YEAR,
    FILINGS_DIR,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Stock Price Download & Caching
# ──────────────────────────────────────────────────────────────────────────────

def get_stock_prices(
    ticker: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """Download daily close prices for *ticker* between *start_date* and
    *end_date* (inclusive).  Results are cached to
    ``MARKET_DATA_DIR / '<ticker>_prices.csv'``.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. ``'JPM'``).
    start_date : str
        ISO date string ``'YYYY-MM-DD'``.
    end_date : str
        ISO date string ``'YYYY-MM-DD'``.

    Returns
    -------
    pd.DataFrame | None
        DataFrame indexed by ``Date`` with a ``Close`` column, or *None*
        if the download fails.
    """
    cache_path = MARKET_DATA_DIR / f"{ticker}_prices.csv"

    # ── Try to load from cache ────────────────────────────────────────────
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
            cached_start = cached.index.min()
            cached_end = cached.index.max()
            req_start = pd.Timestamp(start_date)
            req_end = pd.Timestamp(end_date)

            if cached_start <= req_start and cached_end >= req_end:
                logger.info(
                    "Cache hit for %s (%s → %s)", ticker, start_date, end_date
                )
                return cached.loc[req_start:req_end, ["Close"]]
            else:
                logger.info(
                    "Cache for %s exists but does not cover requested range — "
                    "re-downloading.",
                    ticker,
                )
        except Exception as exc:
            logger.warning("Failed to read cache for %s: %s", ticker, exc)

    # ── Download from Yahoo Finance ───────────────────────────────────────
    try:
        logger.info("Downloading prices for %s (%s → %s)", ticker, start_date, end_date)
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            logger.warning("No price data returned for %s", ticker)
            return None

        # Normalise — yfinance may return multi-level columns for single
        # tickers in newer versions.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Close"]].copy()
        df.index.name = "Date"

        # ── Persist to cache ──────────────────────────────────────────────
        df.to_csv(cache_path)
        logger.info("Cached %d rows for %s → %s", len(df), ticker, cache_path)
        return df

    except Exception as exc:
        logger.error("Failed to download prices for %s: %s", ticker, exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Cumulative Abnormal Return (CAR)
# ──────────────────────────────────────────────────────────────────────────────

def compute_car(
    ticker: str,
    filing_date: str,
    window_before: int,
    window_after: int,
    benchmark: Optional[str] = None,
) -> Optional[float]:
    """Compute the Cumulative Abnormal Return around a filing date.

    CAR = R_stock − R_benchmark

    where R = (close[t+window_after] − close[t−window_before]) / close[t−window_before].

    Parameters
    ----------
    ticker : str
        Company ticker symbol.
    filing_date : str | datetime
        Date the filing was made (``'YYYY-MM-DD'`` or ``datetime``).
    window_before : int
        Trading days before the filing date.
    window_after : int
        Trading days after the filing date.
    benchmark : str, optional
        Benchmark ticker (defaults to ``SECTOR_BENCHMARK`` from config).

    Returns
    -------
    float | None
        CAR value, or *None* if price data is unavailable.
    """
    if benchmark is None:
        benchmark = SECTOR_BENCHMARK

    if isinstance(filing_date, str):
        filing_dt = pd.Timestamp(filing_date)
    else:
        filing_dt = pd.Timestamp(filing_date)

    # Buffer: Provide plenty of days based on window size
    buffer_days = max(30, window_after + 30)
    buf_start = (filing_dt - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    buf_end = (filing_dt + timedelta(days=buffer_days)).strftime("%Y-%m-%d")

    stock_prices = get_stock_prices(ticker, buf_start, buf_end)
    bench_prices = get_stock_prices(benchmark, buf_start, buf_end)

    if stock_prices is None or bench_prices is None:
        logger.warning(
            "CAR: Missing price data for %s or %s around %s",
            ticker,
            benchmark,
            filing_date,
        )
        return None

    try:
        # ── Locate t0: first trading day >= filing_date ───────────────────
        stock_dates = stock_prices.index.sort_values()
        t0_candidates = stock_dates[stock_dates >= filing_dt]
        if t0_candidates.empty:
            logger.warning("No trading day on or after %s for %s", filing_date, ticker)
            return None
        t0 = t0_candidates[0]
        t0_idx = stock_dates.get_loc(t0)

        # ── t_minus and t_plus ────────────────────────────────────────
        # Window parameters are passed as e.g. (-1, 3) which means 1 day before, 3 days after.
        # But wait, config says (-1, 3). If window_before is -1, it means 1 day before. 
        # Or if window_before is 0, it means 0 days before.
        # Let's interpret negative as days *before* t0. So index = t0_idx + window_before.
        t_minus_idx = t0_idx + window_before
        t_plus_idx = t0_idx + window_after

        if t_minus_idx < 0 or t_plus_idx >= len(stock_dates):
            logger.warning(
                "Insufficient trading days around %s for %s (need indices [%d, %d] but bounds are [0, %d])",
                filing_date,
                ticker,
                t_minus_idx,
                t_plus_idx,
                len(stock_dates) - 1,
            )
            return None

        t_minus_1 = stock_dates[t_minus_idx]
        t_plus_3 = stock_dates[t_plus_idx]

        # ── Stock return ──────────────────────────────────────────────────
        close_minus = float(stock_prices.loc[t_minus_1, "Close"])
        close_plus = float(stock_prices.loc[t_plus_3, "Close"])
        r_stock = (close_plus - close_minus) / close_minus

        # ── Benchmark return (same calendar window) ───────────────────────
        bench_dates = bench_prices.index.sort_values()
        # Find closest dates in benchmark
        b_minus_candidates = bench_dates[bench_dates <= t_minus_1]
        b_plus_candidates = bench_dates[bench_dates >= t_plus_3]

        if b_minus_candidates.empty or b_plus_candidates.empty:
            logger.warning(
                "Benchmark %s data does not cover window [%s, %s]",
                benchmark,
                t_minus_1,
                t_plus_3,
            )
            return None

        b_minus_date = b_minus_candidates[-1]
        b_plus_date = b_plus_candidates[0]

        close_b_minus = float(bench_prices.loc[b_minus_date, "Close"])
        close_b_plus = float(bench_prices.loc[b_plus_date, "Close"])
        r_bench = (close_b_plus - close_b_minus) / close_b_minus

        car = r_stock - r_bench
        logger.info(
            "CAR for %s @ %s: %.4f  (R_stock=%.4f, R_bench=%.4f)",
            ticker,
            filing_date,
            car,
            r_stock,
            r_bench,
        )
        return car

    except Exception as exc:
        logger.error("Error computing CAR for %s @ %s: %s", ticker, filing_date, exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 3.  EPS Surprise
# ──────────────────────────────────────────────────────────────────────────────

def get_eps_surprise(ticker: str, year: int) -> Optional[Dict[str, float]]:
    """Retrieve EPS surprise data for *ticker* in the given *year*.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    year : int
        Fiscal year to look up.

    Returns
    -------
    dict | None
        ``{'actual_eps': float, 'estimated_eps': float, 'surprise_pct': float}``
        or *None* when data is unavailable.
    """
    try:
        tkr = yf.Ticker(ticker)

        # `earnings_dates` may not exist for all tickers / years
        if not hasattr(tkr, "earnings_dates") or tkr.earnings_dates is None:
            logger.warning("No earnings_dates attribute for %s", ticker)
            return None

        earnings = tkr.earnings_dates
        if earnings.empty:
            logger.warning("Empty earnings_dates for %s", ticker)
            return None

        # Filter to the requested year
        earnings.index = pd.to_datetime(earnings.index)
        year_mask = earnings.index.year == year
        year_data = earnings.loc[year_mask]

        if year_data.empty:
            logger.info("No earnings dates found for %s in %d", ticker, year)
            return None

        # Pick the most recent row within that year (annual report)
        row = year_data.sort_index(ascending=False).iloc[0]

        est_col = "EPS Estimate"
        act_col = "Reported EPS"

        if est_col not in year_data.columns or act_col not in year_data.columns:
            logger.warning("Missing EPS columns for %s in %d", ticker, year)
            return None

        estimated = row.get(est_col)
        actual = row.get(act_col)

        if pd.isna(estimated) or pd.isna(actual):
            logger.info(
                "EPS values are NaN for %s in %d (est=%s, act=%s)",
                ticker,
                year,
                estimated,
                actual,
            )
            return None

        estimated = float(estimated)
        actual = float(actual)

        if estimated == 0:
            surprise_pct = 0.0
        else:
            surprise_pct = (actual - estimated) / abs(estimated) * 100.0

        logger.info(
            "EPS for %s/%d — actual=%.2f, est=%.2f, surprise=%.1f%%",
            ticker,
            year,
            actual,
            estimated,
            surprise_pct,
        )
        return {
            "actual_eps": actual,
            "estimated_eps": estimated,
            "surprise_pct": surprise_pct,
        }

    except Exception as exc:
        logger.error("Error fetching EPS surprise for %s/%d: %s", ticker, year, exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Helper — Resolve Filing Date
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_filing_date(ticker: str, year: int) -> str:
    """Read the actual conformed filing date from the raw downloaded SEC
    submission headers, falling back to an approximate calendar date if unavailable.

    Parameters
    ----------
    ticker : str
        Company ticker.
    year : int
        Calendar submission year (e.g. 2022 to 2026).

    Returns
    -------
    str
        Filing date as ``'YYYY-MM-DD'``.
    """
    filing_type = COMPANIES.get(ticker, {}).get("filing_type", "10-K")
    base_dir = FILINGS_DIR / "sec-edgar-filings" / ticker / filing_type

    # Attempt 1: Parse actual conformed filed-as-of-date from SEC submission header
    if base_dir.exists():
        for accession_dir in base_dir.iterdir():
            if not accession_dir.is_dir():
                continue
            parts = accession_dir.name.split("-")
            if len(parts) >= 2:
                try:
                    short_year = int(parts[-2])
                    acc_year = 2000 + short_year if short_year < 100 else short_year
                    if acc_year == year:
                        sub_file = accession_dir / "full-submission.txt"
                        if sub_file.exists():
                            with open(sub_file, "r", encoding="utf-8", errors="ignore") as f:
                                for _ in range(100):
                                    line = f.readline()
                                    if not line:
                                        break
                                    if "FILED AS OF DATE:" in line:
                                        date_str = line.split(":")[-1].strip()
                                        if len(date_str) == 8 and date_str.isdigit():
                                            actual_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                                            logger.info("Found actual filing date %s for %s calendar year %d", actual_date, ticker, year)
                                            return actual_date
                except Exception as exc:
                    logger.debug("Failed to extract date from accession %s: %s", accession_dir.name, exc)

    # Attempt 2: Fall back to metadata CSV if present
    meta_path = FILINGS_DIR / ticker / f"{ticker}_filings_metadata.csv"
    if meta_path.exists():
        try:
            meta = pd.read_csv(meta_path)
            if "filing_date" in meta.columns and "year" in meta.columns:
                row = meta.loc[meta["year"] == year]
                if not row.empty:
                    return str(row.iloc[0]["filing_date"])
        except Exception as exc:
            logger.debug("Could not parse metadata for %s: %s", ticker, exc)

    # Attempt 3: Approximate dates within the filing calendar year
    if filing_type == "20-F":
        approx = f"{year}-03-31"  # 20-F filed late March/April
    else:
        approx = f"{year}-02-20"  # 10-K filed mid-to-late February

    logger.info(
        "Using approximate filing date %s for %s/%d (%s)",
        approx,
        ticker,
        year,
        filing_type,
    )
    return approx


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Batch: Fetch All Market Data
# ──────────────────────────────────────────────────────────────────────────────

def fetch_all_market_data(tickers: list[str] | None = None) -> pd.DataFrame:
    """Compute CAR and EPS surprise for every company × year combination
    defined in ``config.COMPANIES``.

    Parameters
    ----------
    tickers : list[str], optional
        If provided, only fetch data for these tickers.

    Returns
    -------
    pd.DataFrame
        Columns: ``[Ticker, Year, Filing_Date, Market_CAR,
        EPS_Actual, EPS_Estimated, EPS_Surprise_Pct]``.
    """
    records: List[Dict] = []

    target_companies = {t: COMPANIES[t] for t in tickers if t in COMPANIES} if tickers else COMPANIES
    for ticker, meta in target_companies.items():
        for year in range(FILING_START_YEAR, FILING_END_YEAR + 1):
            logger.info("─── Processing %s / %d ───", ticker, year)

            filing_date = _resolve_filing_date(ticker, year)

            # Compute CAR for all windows
            car_results = {}
            for wb, wa in CAR_WINDOWS:
                car = compute_car(ticker, filing_date, window_before=wb, window_after=wa, benchmark=meta.get("benchmark"))
                col_name = f"Market_CAR_[{wb},+{wa}]"
                car_results[col_name] = car

            # EPS
            eps = get_eps_surprise(ticker, year)

            record = {
                "Ticker": ticker,
                "Year": year,
                "Filing_Date": filing_date,
                "EPS_Actual": eps["actual_eps"] if eps else None,
                "EPS_Estimated": eps["estimated_eps"] if eps else None,
                "EPS_Surprise_Pct": eps["surprise_pct"] if eps else None,
            }
            record.update(car_results)
            records.append(record)

    df = pd.DataFrame(records)

    out_path = MARKET_DATA_DIR / "market_metrics.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved market metrics (%d rows) → %s", len(df), out_path)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Market Data — SEC Earnings Sentiment Pipeline")
    logger.info("=" * 60)
    result = fetch_all_market_data()
    print(result.to_string(index=False))
