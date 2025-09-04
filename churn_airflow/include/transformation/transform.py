# dags/scripts/transform.py
import os
import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler

PREPARED_DEST = "/usr/local/airflow/dags/data/prepared"
TRANSFORMED_DEST = "/usr/local/airflow/dags/data/transformed"

#BASE_DIR = Path(__file__).resolve().parent.parent.parent  # points to c:\churn_airflow
#PREPARED_DEST = BASE_DIR / "include" / "data" / "prepared"
#TRANSFORMED_DEST = BASE_DIR / "include" / "data" / "transformed"

log_file = Path(os.environ.get("ITRANSFORM_LOG_FILE", "/usr/local/airflow/dags/data/logs/transformation.log"))
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def latest_prepared_file():
    """Get the most recent prepared dataset"""
    prepared = Path(PREPARED_DEST)
    if not prepared.exists():
        raise FileNotFoundError("No prepared data found")
    files = [f for f in prepared.iterdir() if f.suffix == ".csv"]
    if not files:
        raise FileNotFoundError("No prepared data CSV found")
    return max(files, key=os.path.getmtime)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering using actual prepared dataset columns"""
    logging.info(f"Starting feature engineering with shape: {df.shape}")

    # --- Derived / Aggregated Features ---
    if "monthly_spend" in df.columns and "invoices" in df.columns:
        df["avg_monthly_spend_per_invoice"] = (
            df["monthly_spend"] / df["invoices"].replace(0, 1)
        )
        logging.info("Feature created: avg_monthly_spend_per_invoice")

    if set(["sessions_30d", "pages_viewed_30d", "time_on_site_sec_30d"]).issubset(df.columns):
        df["engagement_score"] = (
            df["sessions_30d"] + df["pages_viewed_30d"] + (df["time_on_site_sec_30d"] / 60)
        )
        logging.info("Feature created: engagement_score")

    if "last_seen_at" in df.columns and "sign_up_date" in df.columns:
        df["tenure_proxy"] = df["last_seen_at"] - df["sign_up_date"]
        logging.info("Feature created: tenure_proxy")

    # --- Scaling new features ---
    scaler = StandardScaler()
    new_features = ["avg_monthly_spend_per_invoice", "engagement_score", "tenure_proxy"]
    scale_cols = [c for c in new_features if c in df.columns]

    if scale_cols:
        scaled = scaler.fit_transform(df[scale_cols])
        for i, col in enumerate(scale_cols):
            df[f"{col}_scaled"] = scaled[:, i]
            logging.info(f"Scaled feature created: {col}_scaled")

    logging.info(f"Completed feature engineering. New shape: {df.shape}")

    # Save locally
    Path(TRANSFORMED_DEST).mkdir(parents=True, exist_ok=True)
    out_file = Path(TRANSFORMED_DEST) / "transformed.csv"
    df.to_csv(out_file, index=False)
    logging.info(f"Transformed data saved at {out_file} with shape {df.shape}")

    return df

def generate_sql_schema(table_name="transformed_customers"):
    """Generate SQL schema for the transformed data"""
    schema = f"""
    CREATE TABLE {table_name} (
        customer_id VARCHAR(50) PRIMARY KEY,
        monthly_spend FLOAT,
        invoices INT,
        last_payment_date FLOAT,
        sign_up_date FLOAT,
        last_seen_at FLOAT,
        sessions_30d INT,
        pages_viewed_30d INT,
        time_on_site_sec_30d FLOAT,
        avg_monthly_spend_per_invoice FLOAT,
        engagement_score FLOAT,
        tenure_proxy FLOAT,
        avg_monthly_spend_per_invoice_scaled FLOAT,
        engagement_score_scaled FLOAT,
        tenure_proxy_scaled FLOAT,
        churn INT
    );
    """
    return schema

def sample_queries(table_name="transformed_customers"):
    """Return sample SQL queries"""
    return [
        f"SELECT customer_id, monthly_spend, churn FROM {table_name} WHERE churn=1;",
        f"SELECT AVG(engagement_score) as avg_engagement FROM {table_name} GROUP BY churn;",
        f"SELECT customer_id, tenure_proxy FROM {table_name} ORDER BY tenure_proxy DESC LIMIT 10;"
    ]

def write_summary_md():
    """Write Markdown summary of transformation logic"""
    summary = """# Transformation Summary

## Feature Engineering

### Derived / Aggregated Features
- **avg_monthly_spend_per_invoice** = monthly_spend / invoices
- **engagement_score** = sessions_30d + pages_viewed_30d + (time_on_site_sec_30d / 60)
- **tenure_proxy** = last_seen_at - sign_up_date

### Scaling / Normalization
- Applied **StandardScaler** (mean=0, std=1) on:
  - avg_monthly_spend_per_invoice
  - engagement_score
  - tenure_proxy

### Target
- Data is stored locally as **transformed.csv** inside `/usr/local/airflow/dags/data/transformed`.

---

## Deliverables
- **Transformed dataset**: `transformed.csv`
- **SQL Schema**: `schema.sql`
- **Sample Queries**: `sample_queries.sql`
- **Summary**: `transformation_summary.md`
"""
    summary_file = Path(TRANSFORMED_DEST) / "transformation_summary.md"
    with open(summary_file, "w") as f:
        f.write(summary)
    logging.info(f"Transformation summary saved at {summary_file}")

def transform_data():
    latest_file = latest_prepared_file()
    logging.info(f"Loading prepared dataset {latest_file}")
    df = pd.read_csv(latest_file)
    logging.info(f"Initial dataset shape: {df.shape}")

    transformed_df = feature_engineering(df)

    # Save schema & queries
    Path(TRANSFORMED_DEST).mkdir(parents=True, exist_ok=True)
    schema_file = Path(TRANSFORMED_DEST) / "schema.sql"
    with open(schema_file, "w") as f:
        f.write(generate_sql_schema())

    queries_file = Path(TRANSFORMED_DEST) / "sample_queries.sql"
    with open(queries_file, "w") as f:
        f.write("\n".join(sample_queries()))

    # Write markdown summary
    write_summary_md()

    logging.info(f"✅ Transformation complete. Outputs saved in {TRANSFORMED_DEST}")

if __name__ == "__main__":
    transform_data()
