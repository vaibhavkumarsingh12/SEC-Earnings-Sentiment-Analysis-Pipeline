"""
SEC Earnings Sentiment Analysis Pipeline — Main Orchestrator
=============================================================
End-to-end runner that chains all five phases:
  1. Download SEC filings
  2. Extract MD&A text blocks
  3. Score with FinBERT + Loughran-McDonald
  4. Fetch aligned market data
  5. Statistical analysis & visualization

Usage:
    python main.py                  # Full pipeline
    python main.py --download-only  # Just download filings
    python main.py --extract-only   # Just extract MD&A
    python main.py --score-only     # Just run sentiment scoring
    python main.py --market-only    # Just fetch market data
    python main.py --analyze-only   # Just run analysis on existing data
    python main.py --ticker JPM     # Single company
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

from src.sec_downloader import download_all_filings, find_filing_paths
from src.mda_extractor import extract_mda, clean_mda_text, process_all_filings
from src.text_preprocessor import prepare_chunks
from src.finbert_scorer import FinBERTScorer, score_document
from src.lm_scorer import load_lm_dictionary, score_text
from src.market_data import fetch_all_market_data
from src.analysis import build_master_dataframe, run_correlations, generate_correlation_summary
from src.visualizations import generate_all_plots

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-22s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────
# Phase Runners
# ─────────────────────────────────────────────────────────────────────

def phase_1_download(tickers: list[str] | None = None) -> dict:
    """Phase 1: Download SEC filings from EDGAR."""
    logger.info("═══ PHASE 1: Downloading SEC Filings ═══")
    results = download_all_filings(tickers=tickers)
    total = sum(len(v) for v in results.values())
    logger.info(f"Phase 1 complete — downloaded/found {total} filings across {len(results)} companies")
    return results


def phase_2_extract(tickers: list[str] | None = None) -> dict:
    """Phase 2: Extract MD&A text from downloaded filings."""
    logger.info("═══ PHASE 2: Extracting MD&A Sections ═══")
    results = process_all_filings(tickers=tickers)
    logger.info(f"Phase 2 complete — extracted {len(results)} MD&A sections")
    return results


def phase_3_score(tickers: list[str] | None = None) -> None:
    """Phase 3: Score extracted MD&A text with FinBERT and Loughran-McDonald."""
    logger.info("═══ PHASE 3: Running Sentiment Scoring ═══")

    import pandas as pd

    # Discover available MD&A text files
    mda_files = sorted(config.MDA_TEXTS_DIR.glob("*.txt"))
    if tickers:
        mda_files = [f for f in mda_files if f.stem.split("_")[0] in tickers]

    if not mda_files:
        logger.warning("No MD&A text files found. Run Phase 2 first.")
        return

    # Initialize scorers
    logger.info("Loading FinBERT model...")
    finbert = FinBERTScorer()
    logger.info("Loading Loughran-McDonald dictionary...")
    lm_dict = load_lm_dictionary()

    all_scores = []

    for mda_file in mda_files:
        parts = mda_file.stem.split("_")
        ticker = parts[0]
        year = parts[1] if len(parts) > 1 else "unknown"

        logger.info(f"Scoring {ticker} {year} ({mda_file.name})...")
        text = mda_file.read_text(encoding="utf-8", errors="ignore")

        if len(text.strip()) < 1000:
            logger.warning(f"  Skipping {mda_file.name} — text too short ({len(text)} chars)")
            continue

        # FinBERT scoring
        try:
            finbert_result = score_document(text, scorer=finbert)
            finbert_agg = finbert_result["aggregate"]
        except Exception as e:
            logger.error(f"  FinBERT error for {ticker} {year}: {e}")
            finbert_agg = {
                "net_polarity": None, "positive_ratio": None,
                "negative_ratio": None, "avg_positive": None,
                "avg_negative": None, "avg_neutral": None,
                "dominant_sentiment": None, "num_chunks": 0,
            }

        # Loughran-McDonald scoring
        try:
            lm_result = score_text(text, word_lists=lm_dict)
        except Exception as e:
            logger.error(f"  LM error for {ticker} {year}: {e}")
            lm_result = {
                "lm_net_score": None, "positive_pct": None,
                "negative_pct": None, "uncertainty_pct": None,
                "litigious_pct": None,
            }

        row = {
            "Ticker": ticker,
            "Year": year,
            "FinBERT_Score": finbert_agg.get("net_polarity"),
            "FinBERT_Positive_Ratio": finbert_agg.get("positive_ratio"),
            "FinBERT_Negative_Ratio": finbert_agg.get("negative_ratio"),
            "FinBERT_Avg_Positive": finbert_agg.get("avg_positive"),
            "FinBERT_Avg_Negative": finbert_agg.get("avg_negative"),
            "FinBERT_Avg_Neutral": finbert_agg.get("avg_neutral"),
            "FinBERT_Dominant": finbert_agg.get("dominant_sentiment"),
            "FinBERT_Chunks": finbert_agg.get("num_chunks"),
            "LM_Score": lm_result.get("lm_net_score"),
            "LM_Positive_Pct": lm_result.get("positive_pct"),
            "LM_Negative_Pct": lm_result.get("negative_pct"),
            "LM_Uncertainty_Pct": lm_result.get("uncertainty_pct"),
            "LM_Litigious_Pct": lm_result.get("litigious_pct"),
        }
        all_scores.append(row)
        logger.info(
            f"  FinBERT={finbert_agg.get('net_polarity', 'N/A'):.4f}  "
            f"LM={lm_result.get('lm_net_score', 'N/A'):.6f}  "
            f"Chunks={finbert_agg.get('num_chunks', 0)}"
            if finbert_agg.get("net_polarity") is not None else
            f"  Scoring incomplete for {ticker} {year}"
        )

    # Save all scores
    df_scores = pd.DataFrame(all_scores)
    output_path = config.SENTIMENT_DIR / "all_sentiment_scores.csv"
    df_scores.to_csv(output_path, index=False)
    logger.info(f"Phase 3 complete — saved {len(all_scores)} scores to {output_path}")


def phase_4_market(tickers: list[str] | None = None) -> None:
    """Phase 4: Fetch market data (prices, EPS) from yfinance."""
    logger.info("═══ PHASE 4: Fetching Market Data ═══")
    df = fetch_all_market_data(tickers=tickers)
    logger.info(f"Phase 4 complete — {len(df)} market data records collected")


def phase_5_analyze() -> None:
    """Phase 5: Statistical correlation analysis and visualization."""
    logger.info("═══ PHASE 5: Statistical Analysis & Visualization ═══")

    # Build master dataset
    df = build_master_dataframe()
    if df is None or df.empty:
        logger.error("Master dataset is empty. Ensure Phases 3 and 4 have completed.")
        return

    logger.info(f"Master dataset: {len(df)} rows × {len(df.columns)} columns")

    # Run correlations
    correlations = run_correlations(df)
    generate_correlation_summary(correlations)

    # Generate plots
    plot_paths = generate_all_plots(df)
    logger.info(f"Phase 5 complete — generated {len(plot_paths)} plots")

    # Build dashboard dataset
    try:
        from src.dashboard_builder import build_dashboard_data
        logger.info("═══ Building Dashboard Data JS ═══")
        build_dashboard_data()
    except Exception as e:
        logger.error(f"Failed to build dashboard data.js: {e}")

    # Print final summary table
    _print_summary(df, correlations)


def _print_summary(df, correlations: dict) -> None:
    """Print a formatted summary of the analysis results."""
    print("\n" + "═" * 72)
    print("  SEC EARNINGS SENTIMENT ANALYSIS — RESULTS SUMMARY")
    print("═" * 72)

    print(f"\n  Companies analyzed : {df['Ticker'].nunique()}")
    print(f"  Total filings      : {len(df)}")
    print(f"  Year range         : {df['Year'].min()} – {df['Year'].max()}")

    print("\n  ── Sentiment Score Statistics ──")
    for col in ["FinBERT_Score", "LM_Score"]:
        if col in df.columns and df[col].notna().any():
            print(f"  {col:20s}  mean={df[col].mean():+.4f}  std={df[col].std():.4f}  "
                  f"min={df[col].min():+.4f}  max={df[col].max():+.4f}")

    print("\n  ── Key Correlations ──")
    for pair, vals in correlations.items():
        sig = "✓" if vals.get("significant") else "✗"
        print(f"  {pair:40s}  r={vals.get('pearson_r', 0):+.3f}  "
              f"ρ={vals.get('spearman_rho', 0):+.3f}  "
              f"p={vals.get('pearson_p', 1):.4f}  [{sig}]")

    print("\n" + "═" * 72)
    print(f"  Outputs saved to: {config.OUTPUT_DIR}")
    print("═" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SEC Earnings Sentiment Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", default=True,
                      help="Run the entire pipeline (default)")
    mode.add_argument("--download-only", action="store_true",
                      help="Only download SEC filings")
    mode.add_argument("--extract-only", action="store_true",
                      help="Only extract MD&A sections")
    mode.add_argument("--score-only", action="store_true",
                      help="Only run sentiment scoring")
    mode.add_argument("--market-only", action="store_true",
                      help="Only fetch market data")
    mode.add_argument("--analyze-only", action="store_true",
                      help="Only run analysis on existing data")

    parser.add_argument("--ticker", type=str, default=None,
                        help="Run for a single ticker (e.g., JPM)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [args.ticker.upper()] if args.ticker else None

    if tickers:
        for t in tickers:
            if t not in config.COMPANIES:
                logger.error(f"Unknown ticker: {t}. Available: {list(config.COMPANIES.keys())}")
                sys.exit(1)

    start_time = time.time()
    logger.info("Pipeline starting...")
    logger.info(f"  Target companies: {tickers or list(config.COMPANIES.keys())}")
    logger.info(f"  Year range: {config.FILING_START_YEAR}–{config.FILING_END_YEAR}")
    logger.info(f"  SEC identity: {config.SEC_USER_AGENT}")

    try:
        if args.download_only:
            phase_1_download(tickers)
        elif args.extract_only:
            phase_2_extract(tickers)
        elif args.score_only:
            phase_3_score(tickers)
        elif args.market_only:
            phase_4_market(tickers)
        elif args.analyze_only:
            phase_5_analyze()
        else:
            # Full pipeline
            phase_1_download(tickers)
            phase_2_extract(tickers)
            phase_3_score(tickers)
            phase_4_market(tickers)
            phase_5_analyze()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info(f"Pipeline finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
