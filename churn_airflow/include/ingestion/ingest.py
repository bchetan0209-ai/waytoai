# dags/scripts/ingest.py
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path

RAW_BASE = "/usr/local/airflow/dags/data/raw"
INGEST_DEST = "/usr/local/airflow/dags/data/ingested"

#BASE_DIR = Path(__file__).resolve().parent.parent.parent  # points to c:\churn_airflow
#RAW_BASE = BASE_DIR / "include" / "data" / "raw"
#INGEST_DEST = BASE_DIR / "include" / "data" / "ingested"

log_file = Path(os.environ.get("INGEST_LOG_FILE", "/usr/local/airflow/dags/data/logs/ingestion.log"))
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def ingest_data():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest_dir = Path(INGEST_DEST) / timestamp
    dest_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        Path(RAW_BASE) / "source_csv" / "web_events.csv", # Simulated CSV file
        Path(RAW_BASE) / "source_api" / "subscriptions.json", # Simulated REST API call response
    ]

    for src in sources:
        if src.exists():
            shutil.copy(src, dest_dir / src.name)
            logging.info(f"Ingested {src} -> {dest_dir/src.name}")
        else:
            logging.error(f"Source file missing: {src}")

    logging.info(f"✅ Ingestion complete. Files copied to {dest_dir}")

if __name__ == "__main__":
    ingest_data()
