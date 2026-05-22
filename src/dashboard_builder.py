"""
Dashboard Data Builder — SEC Earnings Sentiment Analysis Pipeline
=================================================================
Reads the master CSV dataset and correlation summary, and compiles them
into a static JavaScript file ('output/dashboard/data.js') to bypass browser CORS
restrictions when the dashboard is opened locally.
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger("dashboard_builder")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-18s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def build_dashboard_data():
    """Reads master_dataset.csv and correlation_summary.csv, and generates data.js."""
    master_csv_path = config.RESULTS_DIR / "master_dataset.csv"
    corr_csv_path = config.RESULTS_DIR / "correlation_summary.csv"
    dashboard_dir = config.OUTPUT_DIR / "dashboard"
    output_js_path = dashboard_dir / "data.js"

    # Create dashboard directory if it doesn't exist
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    if not master_csv_path.exists():
        logger.error(f"Master dataset CSV not found at: {master_csv_path}. Please run the analysis first.")
        return False

    if not corr_csv_path.exists():
        logger.error(f"Correlation summary CSV not found at: {corr_csv_path}. Please run the analysis first.")
        return False

    logger.info("Loading master dataset CSV...")
    df_master = pd.read_csv(master_csv_path)
    
    logger.info("Loading correlation summary CSV...")
    df_corr = pd.read_csv(corr_csv_path)

    # Convert DataFrames to dicts/lists
    master_records = df_master.to_dict(orient="records")
    corr_records = df_corr.to_dict(orient="records")

    # Calculate some helper aggregates to simplify JS logic
    companies_list = df_master["Ticker"].unique().tolist()
    years_list = sorted(df_master["Year"].unique().tolist())
    
    # Build JS file contents
    js_content = f"""/**
 * SEC Earnings Sentiment Analysis Dashboard Data
 * Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
 * Bypasses local CORS restrictions for standard file:/// protocol access.
 */

const MASTER_DATA = {json.dumps(master_records, indent=2)};

const CORRELATION_DATA = {json.dumps(corr_records, indent=2)};

const COMPANIES_CONFIG = {json.dumps(config.COMPANIES, indent=2)};

const AGGREGATE_METADATA = {{
  "generated_at": "{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}",
  "total_records": {len(df_master)},
  "companies": {json.dumps(companies_list)},
  "years": {json.dumps(years_list)},
  "average_finbert_sentiment": {round(df_master["FinBERT_Score"].mean(), 4) if "FinBERT_Score" in df_master.columns else "null"},
  "average_lm_sentiment": {round(df_master["LM_Score"].mean(), 6) if "LM_Score" in df_master.columns else "null"}
}};
"""

    logger.info(f"Writing dashboard data.js to: {output_js_path}")
    output_js_path.write_text(js_content, encoding="utf-8")
    logger.info("Dashboard data successfully built!")
    return True

if __name__ == "__main__":
    build_dashboard_data()
