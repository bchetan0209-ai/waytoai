# dags/scripts/build_feature_store.py
import logging
from pathlib import Path
import pandas as pd
from feature_store import LocalFeatureStore  # if relative import fails, use: from .feature_store import LocalFeatureStore
import os

TRANSFORMED_DEST = Path("/usr/local/airflow/dags/data/transformed")

#BASE_DIR = Path(__file__).resolve().parent.parent.parent  # points to c:\churn_airflow
#TRANSFORMED_DEST = BASE_DIR / "include" / "data" / "transformed"

TRANSFORMED_FILE = None  # auto-pick latest if None
FEATURE_STORE_NAME = "customer_engagement"
FEATURE_STORE_VERSION = "v1"
PRIMARY_KEY = "customer_id"

log_file = Path(os.environ.get("FEATURE_STORE_LOG_FILE", "/usr/local/airflow/dags/data/logs/feature_store.log"))
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def latest_transformed_file() -> Path:
    files = sorted(TRANSFORMED_DEST.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No transformed CSV found in {TRANSFORMED_DEST}")
    return files[-1]

def main():
    csv_path = Path(TRANSFORMED_FILE) if TRANSFORMED_FILE else latest_transformed_file()
    logging.info(f"Loading transformed dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    # Choose the features to publish to the store (adjust as needed)
    # Keep scaled + unscaled engineered features for flexibility
    candidates = [
        "avg_monthly_spend_per_invoice",
        "avg_monthly_spend_per_invoice_scaled",
        "engagement_score",
        "engagement_score_scaled",
        "tenure_proxy",
        "tenure_proxy_scaled"
    ]

    # Filter to only those present
    keep = [c for c in candidates if c in df.columns]
    if not keep:
        raise ValueError("No engineered features found. Ensure transform.py created them.")

    fs = LocalFeatureStore()
    out_path = fs.materialize_from_dataframe(
        df=df,
        name=FEATURE_STORE_NAME,
        version=FEATURE_STORE_VERSION,
        primary_key=PRIMARY_KEY,
        keep_columns=keep,
        description="Customer engagement & value engineered features (v1).",
        source="dags/scripts/transform.py",
    )
    logging.info(f"Materialized to: {out_path}")

if __name__ == "__main__":
    main()
