# dags/scripts/prepare.py
import os
import pandas as pd
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

INGEST_DEST = "/usr/local/airflow/dags/data/ingested" 
PREPARED_DEST = "/usr/local/airflow/dags/data/prepared"
EDA_DEST = "/usr/local/airflow/dags/data/eda"

#BASE_DIR = Path(__file__).resolve().parent.parent.parent  # points to c:\churn_airflow
#INGEST_DEST = BASE_DIR / "include" / "data" / "ingested"
#PREPARED_DEST = BASE_DIR / "include" / "data" / "prepared"
#EDA_DEST = BASE_DIR / "include" / "data" / "eda"

log_file = Path(os.environ.get("PREPARE_LOG_FILE", "/usr/local/airflow/dags/data/logs/preparation.log"))
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def latest_ingested_folder():
    """Get the most recent ingested batch folder"""
    ingested = Path(INGEST_DEST)
    if not ingested.exists():
        raise FileNotFoundError("No ingested data found")
    folders = [f for f in ingested.iterdir() if f.is_dir()]
    if not folders:
        raise FileNotFoundError("No ingested batch folders found")
    return max(folders, key=os.path.getmtime)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Handle missing values: fill numeric with median, categorical with mode
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Save churn separately to avoid transformations
    churn_series = None
    if "churn" in df.columns:
        churn_series = df["churn"]
        df = df.drop(columns=["churn"])

    # Save customer_id separately to avoid transformations
    customer_id_series = None
    if "customer_id" in df.columns:
        customer_id_series = df["customer_id"]
        df = df.drop(columns=["customer_id"])

    # Encode categorical variables
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].nunique() <= 5:  # small cardinality → one-hot encode
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1).drop(columns=[col])
        else:  # high cardinality → label encode
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Standardize numerical attributes (exclude customer_id + churn)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Re-attach id + churn columns as-is
    if customer_id_series is not None:
        df["customer_id"] = customer_id_series.values
    if churn_series is not None:
        df["churn"] = churn_series.values

    return df

def perform_eda(df: pd.DataFrame, eda_dir: Path):
    """Generate summary statistics and visualizations"""
    eda_dir.mkdir(parents=True, exist_ok=True)

    # Summary statistics
    df.describe(include="all").to_csv(eda_dir / "summary_stats.csv")

    # Histograms (exclude churn)
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if col == "churn":
            continue
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram of {col}")
        plt.savefig(eda_dir / f"hist_{col}.png")
        plt.close()

    # Boxplots (exclude churn)
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if col == "churn":
            continue
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(eda_dir / f"box_{col}.png")
        plt.close()

    # Correlation heatmap (exclude churn)
    corr_df = df.drop(columns=["churn"], errors="ignore")
    if not corr_df.empty:
        plt.figure(figsize=(10,8))
        sns.heatmap(corr_df.corr(), annot=False, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(eda_dir / "correlation_heatmap.png")
        plt.close()

    # --- Churn-specific EDA ---
    if "churn" in df.columns:
        # Distribution of churn values
        plt.figure()
        sns.countplot(x=df["churn"])
        plt.title("Churn Distribution")
        plt.savefig(eda_dir / "churn_distribution.png")
        plt.close()

        # If categorical features exist, show churn vs. each
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            plt.figure(figsize=(6,4))
            sns.countplot(x=col, hue="churn", data=df)
            plt.title(f"Churn by {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(eda_dir / f"churn_by_{col}.png")
            plt.close()

def prepare_data():
    latest_folder = latest_ingested_folder()
    logging.info(f"Preparing data from {latest_folder}")

    # Load ingested files
    web_events_path = latest_folder / "web_events.csv"
    subs_path = latest_folder / "subscriptions.json"
    print(f"Loading files: {web_events_path}, {subs_path}")
    if not web_events_path.exists() or not subs_path.exists():
        raise FileNotFoundError("Missing required ingested files (web_events.csv, subscriptions.json)")

    web_df = pd.read_csv(web_events_path)
    subs_df = pd.read_json(subs_path)

    # --- Ensure churn column ---
    if "is_churn" in web_df.columns:
        web_df.rename(columns={"is_churn": "churn"}, inplace=True)
    elif "churn" in web_df.columns:
        logging.info("'churn' column already present, keeping as is.")
    else:
        logging.warning("No 'is_churn' or 'churn' column found. Setting churn=0 for all.")
        web_df["churn"] = 0

    # --- Join on customer_id ---
    if "customer_id" in web_df.columns and "customer_id" in subs_df.columns:
        merged_df = pd.merge(subs_df, web_df, on="customer_id", how="inner")
    else:
        logging.warning("No common customer_id column found. Concatenating instead.")
        merged_df = pd.concat([subs_df, web_df], axis=1)

    # Preprocess data
    prepared_df = preprocess_data(merged_df)

    # Perform EDA
    perform_eda(prepared_df, Path(EDA_DEST))

    # Save prepared data
    Path(PREPARED_DEST).mkdir(parents=True, exist_ok=True)
    prepared_file = Path(PREPARED_DEST) / f"prepared_{latest_folder.name}.csv"
    prepared_df.to_csv(prepared_file, index=False)

    logging.info(f"✅ Data preparation complete. Prepared dataset saved at {prepared_file}")

if __name__ == "__main__":
    prepare_data()
