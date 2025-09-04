# dags/scripts/validate.py
import os
import pandas as pd
import logging
from pathlib import Path

INGEST_DEST = "/usr/local/airflow/dags/data/ingested"
REPORTS_DIR = "/usr/local/airflow/dags/reports"

#BASE_DIR = Path(__file__).resolve().parent.parent.parent  # points to c:\churn_airflow
#REPORTS_DIR = BASE_DIR / "include" / "reports"
#INGEST_DEST = BASE_DIR / "include" / "data" / "ingested"

log_file = Path(os.environ.get("VALIDATE_LOG_FILE", "/usr/local/airflow/dags/data/logs/validation.log"))
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

def validate_csv(file_path):
    df = pd.read_csv(file_path)
    issues = {}

    # Check for missing values
    missing = df.isnull().sum()
    issues["missing"] = missing[missing > 0].to_dict()

    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues["duplicates"] = duplicates

    # Basic schema validation (example)
    issues["columns"] = list(df.columns)

    return issues

def validate_json(file_path):
    #df = pd.read_json(file_path)
    issues = {}
    print(f"Validating JSON file: {file_path}")
    try:
        df = pd.read_json(file_path)
    except ValueError as e:
        issues["error"] = f"Invalid JSON: {e}"
        return issues
    except Exception as e:
        issues["error"] = f"Error reading JSON: {e}"
        return issues

    

    missing = df.isnull().sum()
    issues["missing"] = missing[missing > 0].to_dict()

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues["duplicates"] = duplicates

    issues["columns"] = list(df.columns)

    return issues

def run_validation():
    latest_folder = latest_ingested_folder()
    logging.info(f"Validating files in {latest_folder}")

    all_issues = {}

    for file in latest_folder.iterdir():
        if file.suffix == ".csv":
            all_issues[file.name] = validate_csv(file)
        elif file.suffix == ".json":
            all_issues[file.name] = validate_json(file)
        else:
            logging.warning(f"Skipping unsupported file: {file}")

    # Save report
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    report_file = Path(REPORTS_DIR) / f"data_quality_report_{latest_folder.name}.csv"
    pd.DataFrame(all_issues).to_csv(report_file)

    logging.info(f"✅ Data validation complete. Report saved at {report_file}")

if __name__ == "__main__":
    run_validation()
