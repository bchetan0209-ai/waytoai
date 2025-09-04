
"""
Astro DAG (Airflow 2.7+/3.x compatible): churn_pipeline_astro_v5
-----------------------------------------------------------------
- Uses `schedule=` (not `schedule_interval`) for Airflow 2.7+/3.x.
- Keeps provider imports inside task callables (import-safe).
- Expects your pipeline code under Astro project **include/** (NOT under dags/).
  Astro maps that to: /usr/local/airflow/include

Project layout example:
  dags/
    churn_pipeline_astro_v5.py
  include/
    ingestion/ingest.py
    preparation/prepare.py
    transformation/...
    modeling/...
    evaluation/...
    deployment/...
    data/raw/...

This prevents Airflow from importing your helper scripts as DAGs.
"""

from __future__ import annotations

import os
import glob
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from pendulum import timezone

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

# ------------------------------------------------------------------
# Paths (align with your scripts' constants)
# ------------------------------------------------------------------
INCLUDE_DIR = "/usr/local/airflow/include"
RAW_SRC_DIR = os.path.join(INCLUDE_DIR, "data", "raw")

RAW_BASE = "/usr/local/airflow/dags/data/raw"
INGEST_DEST = "/usr/local/airflow/dags/data/ingested"
PREPARED_DEST = "/usr/local/airflow/dags/data/prepared"
TRANSFORMED_DEST = "/usr/local/airflow/dags/data/transformed"
MODELS_DIR = "/usr/local/airflow/dags/data/models"
EVAL_DIR = "/usr/local/airflow/dags/data/evaluation"
EDA_DIR = "/usr/local/airflow/dags/data/eda"

POSTGRES_CONN_ID = "postgres_default"
TARGET_TABLE = "churn_features"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def ensure_dirs():
    for d in [RAW_BASE, INGEST_DEST, PREPARED_DEST, TRANSFORMED_DEST, MODELS_DIR, EVAL_DIR, EDA_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def seed_raw_from_include():
    """Copy raw files from include/data/raw -> /usr/local/airflow/dags/data/raw"""
    if not os.path.exists(RAW_SRC_DIR):
        raise FileNotFoundError(f"Expected raw source at {RAW_SRC_DIR}")
    ensure_dirs()
    for root, _, files in os.walk(RAW_SRC_DIR):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, RAW_SRC_DIR)
            dst = os.path.join(RAW_BASE, rel)
            Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)


def _latest_file(directory: str, pattern: str = "*.csv") -> Optional[str]:
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

# ------------------------------------------------------------------
# DAG
# ------------------------------------------------------------------
default_args = {
    "owner": "astro",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

local_tz = timezone("Asia/Kolkata")

with DAG(
    dag_id="churn_pipeline_astro_v5",
    description="Astro: churn pipeline (Airflow 2.7+/3.x compatible)",
    default_args=default_args,
    start_date=datetime(2025, 8, 1, tzinfo=local_tz),
    schedule="45 21 * * *",   # Every day at 9:45 PM IST
    catchup=False,
    max_active_runs=1,
    tags=["astro", "churn", "ml", "etl"],
) as dag:

    t_make_dirs = PythonOperator(
        task_id="make_data_dirs",
        python_callable=ensure_dirs,
    )

    t_seed_raw = PythonOperator(
        task_id="seed_raw_from_include",
        python_callable=seed_raw_from_include,
    )

    py = "python -u"

    # Add INCLUDE_DIR to PYTHONPATH so intra-package imports in your scripts succeed
    env_with_path = {"PYTHONPATH": f"{INCLUDE_DIR}:" + os.environ.get("PYTHONPATH", "")}

    #t_ingest = BashOperator(
    #    task_id="ingest",
    #    bash_command=f"{py} {INCLUDE_DIR}/ingestion/ingest.py",
    #    env=env_with_path,
    #)

    py = "python -u"
    env_with_path = {"PYTHONPATH": f"{INCLUDE_DIR}:" + os.environ.get("PYTHONPATH", "")}

    t_ingest = BashOperator(
        task_id="ingest",
        bash_command=f"{py} {INCLUDE_DIR}/ingestion/ingest.py",
        env=env_with_path,
    )

    t_prepare = BashOperator(
        task_id="prepare",
        bash_command=f"{py} {INCLUDE_DIR}/preparation/prepare.py",
        env=env_with_path,
    )

    t_validate = BashOperator(
        task_id="validate",
        bash_command=f"{py} {INCLUDE_DIR}/validation/validate.py",
        env=env_with_path,
    )

    t_transform = BashOperator(
        task_id="transform",
        bash_command=f"{py} {INCLUDE_DIR}/transformation/transform.py",
        env=env_with_path,
    )

    t_build_feature_store = BashOperator(
        task_id="build_feature_store",
        bash_command=f"{py} {INCLUDE_DIR}/transformation/build_feature_store.py",
        env=env_with_path,
    )

    t_train = BashOperator(
        task_id="train_model",
        bash_command=f"{py} {INCLUDE_DIR}/modeling/train.py",
        env=env_with_path,
    )

    t_evaluate = BashOperator(
        task_id="evaluate_model",
        bash_command=f"{py} {INCLUDE_DIR}/evaluation/evaluate.py",
        env=env_with_path,
    )

    t_deploy = BashOperator(
        task_id="deploy_model",
        bash_command=f"{py} {INCLUDE_DIR}/deployment/deploy.py",
        env=env_with_path,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    t_make_dirs >> t_seed_raw >> t_ingest >> t_prepare >> t_validate >> t_transform
    t_transform >> t_build_feature_store >> t_train >> t_evaluate >> t_deploy
