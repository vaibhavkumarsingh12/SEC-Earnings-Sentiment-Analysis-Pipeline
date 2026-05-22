"""
Visualization Module — SEC Earnings Sentiment Analysis Pipeline
================================================================
Generates publication-quality plots that relate sentiment scores to
market performance metrics (CAR, EPS surprise).
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

from config import PLOTS_DIR, PLOT_DPI, PLOT_STYLE, PLOT_PALETTE, FIGURE_SIZE, RESULTS_DIR, CAR_WINDOWS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Module-Level Style Configuration
# ---------------------------------------------------------------------------
sns.set_style("darkgrid")
sns.set_palette(PLOT_PALETTE)

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "figure.figsize": FIGURE_SIZE,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,          # screen DPI (file DPI set at save time)
        "savefig.dpi": PLOT_DPI,
        "savefig.bbox": "tight",
    }
)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Sentiment vs CAR  (scatter + regression)
# ──────────────────────────────────────────────────────────────────────────────

def plot_sentiment_vs_car(
    df: pd.DataFrame,
    score_column: str = "FinBERT_Score",
    car_column: str = "Market_CAR_[-1,+3]",
    title: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Scatter plot of sentiment score vs Cumulative Abnormal Return with a
    regression line and 95 % confidence band.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataset with *score_column* and *car_column*.
    score_column : str
        Column to plot on the x-axis (default ``'FinBERT_Score'``).
    car_column : str
        Column to plot on the y-axis (default ``'Market_CAR_[-1,+3]'``).
    title : str, optional
        Custom plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plot_df = df[[score_column, car_column, "Ticker"]].dropna()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Regression + CI band
    sns.regplot(
        data=plot_df,
        x=score_column,
        y=car_column,
        scatter=False,
        ci=95,
        line_kws={"color": "#2c3e50", "linewidth": 2},
        ax=ax,
    )

    # Scatter — colour by Ticker
    sns.scatterplot(
        data=plot_df,
        x=score_column,
        y=car_column,
        hue="Ticker",
        style="Ticker",
        s=100,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    # Annotate with Pearson r
    if len(plot_df) >= 3:
        r, p = pearsonr(plot_df[score_column], plot_df[car_column])
        annotation = f"Pearson r = {r:.3f}\np = {p:.3f}"
        ax.annotate(
            annotation,
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85),
        )

    ax.set_xlabel("Net Polarity Score")
    ax.set_ylabel(f"Cumulative Abnormal Return {car_column.split('_')[-1]} (%)")
    ax.set_title(title or f"{score_column} vs {car_column}")
    ax.legend(title="Ticker", loc="lower right")

    out_path = PLOTS_DIR / f"{score_column}_vs_CAR.png"
    fig.savefig(out_path, dpi=PLOT_DPI)
    logger.info("Saved → %s", out_path)
    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Sentiment vs EPS Surprise  (scatter + regression)
# ──────────────────────────────────────────────────────────────────────────────

def plot_sentiment_vs_eps(
    df: pd.DataFrame,
    score_column: str = "FinBERT_Score",
    title: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Scatter plot of sentiment score vs EPS Surprise percentage.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataset.
    score_column : str
        Sentiment column for the x-axis.
    title : str, optional
        Custom plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plot_df = df[[score_column, "EPS_Surprise_Pct", "Ticker"]].dropna()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    sns.regplot(
        data=plot_df,
        x=score_column,
        y="EPS_Surprise_Pct",
        scatter=False,
        ci=95,
        line_kws={"color": "#2c3e50", "linewidth": 2},
        ax=ax,
    )

    sns.scatterplot(
        data=plot_df,
        x=score_column,
        y="EPS_Surprise_Pct",
        hue="Ticker",
        style="Ticker",
        s=100,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    if len(plot_df) >= 3:
        r, p = pearsonr(plot_df[score_column], plot_df["EPS_Surprise_Pct"])
        annotation = f"Pearson r = {r:.3f}\np = {p:.3f}"
        ax.annotate(
            annotation,
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85),
        )

    ax.set_xlabel("Net Polarity Score")
    ax.set_ylabel("EPS Surprise (%)")
    ax.set_title(title or f"{score_column} vs EPS Surprise")
    ax.legend(title="Ticker", loc="lower right")

    out_path = PLOTS_DIR / f"{score_column}_vs_EPS.png"
    fig.savefig(out_path, dpi=PLOT_DPI)
    logger.info("Saved → %s", out_path)
    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Comparative Bar Chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparative_bar(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Grouped bar chart comparing FinBERT and Loughran-McDonald sentiment
    scores for each Ticker × Year combination.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataset with ``FinBERT_Score`` and ``LM_Score``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plot_df = df[["Ticker", "Year", "FinBERT_Score", "LM_Score"]].dropna()
    plot_df = plot_df.copy()
    plot_df["Label"] = plot_df["Ticker"] + "_" + plot_df["Year"].astype(str)

    melted = plot_df.melt(
        id_vars="Label",
        value_vars=["FinBERT_Score", "LM_Score"],
        var_name="Method",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(max(FIGURE_SIZE[0], len(plot_df) * 0.8), FIGURE_SIZE[1]))

    sns.barplot(
        data=melted,
        x="Label",
        y="Score",
        hue="Method",
        palette=["#3498db", "#e74c3c"],
        edgecolor="white",
        ax=ax,
    )

    ax.set_xlabel("Company – Year")
    ax.set_ylabel("Sentiment Score")
    ax.set_title("Comparative Sentiment Scores: FinBERT vs Loughran-McDonald")
    ax.legend(title="Method")
    plt.xticks(rotation=45, ha="right")

    out_path = PLOTS_DIR / "comparative_sentiment_scores.png"
    fig.savefig(out_path, dpi=PLOT_DPI)
    logger.info("Saved → %s", out_path)
    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Correlation Heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Annotated correlation heatmap of all numeric columns in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataset.

    Returns
    -------
    matplotlib.figure.Figure
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax,
    )

    ax.set_title("Correlation Matrix — Sentiment & Market Metrics")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    out_path = PLOTS_DIR / "correlation_heatmap.png"
    fig.savefig(out_path, dpi=PLOT_DPI)
    logger.info("Saved → %s", out_path)
    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CAR Window Expansion PEAD Plot
# ──────────────────────────────────────────────────────────────────────────────

from scipy.stats import spearmanr

def plot_car_window_expansion(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Line plot showing how the Spearman rho correlation between FinBERT_Score 
    and CAR evolves as the event window expands (PEAD effect).
    """
    windows = []
    rhos = []
    labels = []

    for wb, wa in CAR_WINDOWS:
        car_col = f"Market_CAR_[{wb},+{wa}]"
        if car_col in df.columns:
            plot_df = df[["FinBERT_Score", car_col]].dropna()
            if len(plot_df) >= 3:
                rho, p = spearmanr(plot_df["FinBERT_Score"], plot_df[car_col])
                rhos.append(rho)
                windows.append(wa)
                labels.append(f"[{wb}, +{wa}]")

    if not rhos:
        logger.warning("No CAR window data available for PEAD plot.")
        return plt.figure()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Plot line
    ax.plot(windows, rhos, marker='o', linestyle='-', linewidth=2.5, markersize=8, color="#2980b9")
    
    # Highlight points above zero
    for i, rho in enumerate(rhos):
        color = "green" if rho > 0 else "red"
        ax.plot(windows[i], rhos[i], marker='o', markersize=10, color=color)
        
        # Annotate
        ax.annotate(
            f"ρ = {rho:.3f}",
            (windows[i], rhos[i]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

    # Set x-ticks to the exact window sizes
    ax.set_xticks(windows)
    ax.set_xticklabels(labels)

    ax.set_xlabel("Event Window Horizon (Days)")
    ax.set_ylabel("Spearman Correlation (ρ)")
    ax.set_title("Post-Earnings Announcement Drift: Sentiment Predictability over Time")
    
    # Add a zero-line for reference
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)

    out_path = PLOTS_DIR / "FinBERT_CAR_Window_Expansion.png"
    fig.savefig(out_path, dpi=PLOT_DPI)
    logger.info("Saved → %s", out_path)
    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Generate All Plots
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_plots(df: pd.DataFrame) -> List[str]:
    """Run every plot function and return a list of saved file paths.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataset.

    Returns
    -------
    list[str]
        Absolute paths of all generated plot images.
    """
    saved: List[str] = []

    # FinBERT vs CAR (baseline window)
    plot_sentiment_vs_car(df, score_column="FinBERT_Score", car_column="Market_CAR_[-1,+3]")
    saved.append(str(PLOTS_DIR / "FinBERT_Score_vs_CAR.png"))

    # LM vs CAR (baseline window)
    plot_sentiment_vs_car(df, score_column="LM_Score", car_column="Market_CAR_[-1,+3]")
    saved.append(str(PLOTS_DIR / "LM_Score_vs_CAR.png"))

    # FinBERT vs EPS
    plot_sentiment_vs_eps(df, score_column="FinBERT_Score")
    saved.append(str(PLOTS_DIR / "FinBERT_Score_vs_EPS.png"))

    # LM vs EPS
    plot_sentiment_vs_eps(df, score_column="LM_Score")
    saved.append(str(PLOTS_DIR / "LM_Score_vs_EPS.png"))

    # Comparative bar
    plot_comparative_bar(df)
    saved.append(str(PLOTS_DIR / "comparative_sentiment_scores.png"))

    # Correlation heatmap
    plot_correlation_heatmap(df)
    saved.append(str(PLOTS_DIR / "correlation_heatmap.png"))
    
    # CAR Window Expansion
    plot_car_window_expansion(df)
    saved.append(str(PLOTS_DIR / "FinBERT_CAR_Window_Expansion.png"))

    logger.info("Generated %d plots:", len(saved))
    for p in saved:
        logger.info("  • %s", p)

    return saved


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Visualizations — SEC Earnings Sentiment Pipeline")
    logger.info("=" * 60)

    master_path = RESULTS_DIR / "master_dataset.csv"
    if not master_path.exists():
        logger.error("Master dataset not found at %s — run analysis.py first.", master_path)
        sys.exit(1)

    master_df = pd.read_csv(master_path)
    paths = generate_all_plots(master_df)

    print(f"\n✓ {len(paths)} plots saved to {PLOTS_DIR}")
