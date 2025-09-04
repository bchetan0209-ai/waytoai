#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
import logging
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Setup directories
DATA_DIR = "/usr/local/airflow/dags/data/transformed"
MODEL_DIR = "/usr/local/airflow/dags/data/models"

#DATA_DIR = Path("include/data/transformed")
#MODEL_DIR = Path("include/data/models")

for d in [Path(MODEL_DIR)]:
    d.mkdir(parents=True, exist_ok=True)

log_file = Path(os.environ.get("TRAIN_LOG_FILE", "/usr/local/airflow/dags/data/logs/train.log"))
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# -------------------------------
# Helpers
# -------------------------------
def _latest_csv(folder: Path) -> Path:
    folder = Path(folder)  # <-- key fix
    if not folder.exists():
        raise FileNotFoundError(f"Processed folder not found: {folder}")
    files = sorted(folder.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    return files[-1]

def load_data():
    csv_path = _latest_csv(DATA_DIR)
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {csv_path} shape={df.shape}")
    if "churn" not in df.columns:
        raise ValueError("Dataset must include churn column.")
    X = df.drop(columns=["churn", "customer_id"], errors="ignore")
    y = df["churn"].astype(int)
    return X, y

def train_models(X_train, y_train):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    }
    if XGB_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            use_label_encoder=False, eval_metric="logloss"
        )
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        logging.info(f"Trained {name}")
    return trained

def save_models(models, version_ts):
    saved_paths = {}
    for name, model in models.items():
        out_path = Path(MODEL_DIR) / f"{name}_{version_ts}.pkl"
        joblib.dump(model, out_path)
        saved_paths[name] = str(out_path)
        logging.info(f"Saved {name} -> {out_path}")
    return saved_paths

# -------------------------------
# Main
# -------------------------------
def main():
    version_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    X, y = load_data()

    # --- Save feature list BEFORE splitting ---
    feature_list = list(X.columns)
    feature_list_path = Path(MODEL_DIR) / f"features_{version_ts}.json"
    with open(feature_list_path, "w") as f:
        json.dump(feature_list, f)
    logging.info(f"Saved feature list -> {feature_list_path}")

    # Train/test split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = train_models(X_train, y_train)
    save_models(models, version_ts)

    print(f"✅ Training complete. Models + feature list saved in {MODEL_DIR}")

if __name__ == "__main__":
    main()
