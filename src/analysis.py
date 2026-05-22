"""
Statistical Analysis Module — SEC Earnings Sentiment Analysis Pipeline
======================================================================
Merges sentiment scores with market metrics, computes Pearson and Spearman
correlations, and generates summary tables.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

from config import RESULTS_DIR, SENTIMENT_DIR, MARKET_DATA_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

# ──────────────────────────────────────────────────────────────────────────────
# Correlation Pair Definitions
# ──────────────────────────────────────────────────────────────────────────────

CORRELATION_PAIRS: List[Tuple[str, str]] = [
    ("FinBERT_Score", "Market_CAR"),
    ("LM_Score", "Market_CAR"),
    ("FinBERT_Score", "EPS_Surprise_Pct"),
    ("LM_Score", "EPS_Surprise_Pct"),
]

SIGNIFICANCE_THRESHOLD = 0.05


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Build Master DataFrame
# ──────────────────────────────────────────────────────────────────────────────

def build_master_dataframe(
    sentiment_file: Optional[str] = None,
    market_file: Optional[str] = None,
) -> pd.DataFrame:
    """Merge sentiment scores with market metrics into a single dataset.

    Parameters
    ----------
    sentiment_file : str or Path, optional
        Path to the sentiment CSV.  Defaults to
        ``SENTIMENT_DIR / 'all_sentiment_scores.csv'``.
    market_file : str or Path, optional
        Path to the market metrics CSV.  Defaults to
        ``MARKET_DATA_DIR / 'market_metrics.csv'``.

    Returns
    -------
    pd.DataFrame
        Merged dataset saved to ``RESULTS_DIR / 'master_dataset.csv'``.
    """
    # ── Resolve paths ─────────────────────────────────────────────────────
    sentiment_path = (
        Path(sentiment_file) if sentiment_file
        else SENTIMENT_DIR / "all_sentiment_scores.csv"
    )
    market_path = (
        Path(market_file) if market_file
        else MARKET_DATA_DIR / "market_metrics.csv"
    )

    logger.info("Loading sentiment data from  %s", sentiment_path)
    logger.info("Loading market data from     %s", market_path)

    sentiment_df = pd.read_csv(sentiment_path)
    market_df = pd.read_csv(market_path)

    # ── Merge on Ticker + Year ────────────────────────────────────────────
    merged = pd.merge(sentiment_df, market_df, on=["Ticker", "Year"], how="inner")
    logger.info(
        "Merged dataset: %d rows  (%d sentiment × %d market → %d matched)",
        len(merged),
        len(sentiment_df),
        len(market_df),
        len(merged),
    )

    # ── Persist ───────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "master_dataset.csv"
    merged.to_csv(out_path, index=False)
    logger.info("Saved master dataset → %s", out_path)

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Run Correlations
# ──────────────────────────────────────────────────────────────────────────────

def run_correlations(df: pd.DataFrame) -> Dict[str, Dict]:
    """Compute Pearson and Spearman correlations for pre-defined pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataset containing sentiment and market columns.

    Returns
    -------
    dict
        Nested dict keyed by ``'X vs Y'`` with sub-keys
        ``pearson_r``, ``pearson_p``, ``spearman_rho``, ``spearman_p``,
        ``significant``.
    """
    results: Dict[str, Dict] = {}

    for x_col, y_col in CORRELATION_PAIRS:
        label = f"{x_col} vs {y_col}"

        # Drop rows with NaN in either column
        pair_df = df[[x_col, y_col]].dropna()

        if len(pair_df) < 3:
            logger.warning(
                "Skipping %s — only %d valid observations", label, len(pair_df)
            )
            results[label] = {
                "pearson_r": None,
                "pearson_p": None,
                "spearman_rho": None,
                "spearman_p": None,
                "significant": False,
                "n": len(pair_df),
            }
            continue

        x = pair_df[x_col].values
        y = pair_df[y_col].values

        p_r, p_p = pearsonr(x, y)
        s_rho, s_p = spearmanr(x, y)

        significant = (p_p < SIGNIFICANCE_THRESHOLD) or (s_p < SIGNIFICANCE_THRESHOLD)

        results[label] = {
            "pearson_r": round(p_r, 4),
            "pearson_p": round(p_p, 4),
            "spearman_rho": round(s_rho, 4),
            "spearman_p": round(s_p, 4),
            "significant": significant,
            "n": len(pair_df),
        }

    # ── Pretty-print summary ──────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 90)
    logger.info("  CORRELATION SUMMARY")
    logger.info("=" * 90)
    logger.info(
        "%-35s  %8s  %8s  %8s  %8s  %4s  %3s",
        "Pair", "Pears_r", "Pears_p", "Spear_ρ", "Spear_p", "Sig?", "N",
    )
    logger.info("-" * 90)
    for label, vals in results.items():
        sig_marker = " ✓ " if vals["significant"] else "   "
        logger.info(
            "%-35s  %8s  %8s  %8s  %8s  %4s  %3d",
            label,
            _fmt(vals["pearson_r"]),
            _fmt(vals["pearson_p"]),
            _fmt(vals["spearman_rho"]),
            _fmt(vals["spearman_p"]),
            sig_marker,
            vals["n"],
        )
    logger.info("=" * 90)

    return results


def _fmt(val: Optional[float]) -> str:
    """Format a float for the summary table, returning '—' for None."""
    if val is None:
        return "    —   "
    return f"{val:8.4f}"


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Generate Correlation Summary
# ──────────────────────────────────────────────────────────────────────────────

def generate_correlation_summary(correlations: Dict[str, Dict]) -> pd.DataFrame:
    """Convert the correlations dict into a presentable DataFrame and save it.

    Parameters
    ----------
    correlations : dict
        Output of :func:`run_correlations`.

    Returns
    -------
    pd.DataFrame
        Columns: ``[Pair, Pearson_r, Pearson_p, Spearman_rho, Spearman_p,
        Significant, N]``.
    """
    rows = []
    for pair, vals in correlations.items():
        rows.append(
            {
                "Pair": pair,
                "Pearson_r": vals["pearson_r"],
                "Pearson_p": vals["pearson_p"],
                "Spearman_rho": vals["spearman_rho"],
                "Spearman_p": vals["spearman_p"],
                "Significant": vals["significant"],
                "N": vals["n"],
            }
        )

    summary_df = pd.DataFrame(rows)

    out_path = RESULTS_DIR / "correlation_summary.csv"
    summary_df.to_csv(out_path, index=False)
    logger.info("Saved correlation summary → %s", out_path)

    return summary_df


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Statistical Analysis — SEC Earnings Sentiment Pipeline")
    logger.info("=" * 60)

    master = build_master_dataframe()
    correlations = run_correlations(master)
    summary = generate_correlation_summary(correlations)

    print("\n", summary.to_string(index=False))
