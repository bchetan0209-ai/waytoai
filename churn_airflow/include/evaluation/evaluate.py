#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

DATA_DIR = "/usr/local/airflow/dags/data/transformed"
MODEL_DIR = "/usr/local/airflow/dags/data/models"
REPORT_DIR = "/usr/local/airflow/dags/data/evaluation"

#DATA_DIR = Path("include/data/transformed")
#MODEL_DIR = Path("include/data/models")
#REPORT_DIR = Path("include/data/evaluation")

for d in [Path(REPORT_DIR)]:
    d.mkdir(parents=True, exist_ok=True)

log_file = Path(os.environ.get("EVALUATE_LOG_FILE", "/usr/local/airflow/dags/data/logs/evaluation.log"))
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def _latest_csv(folder: Path) -> Path:
    folder = Path(folder)  # <-- key fix
    if not folder.exists():
        raise FileNotFoundError(f"Processed folder not found: {folder}")
    
    files = sorted(folder.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    return files[-1]

def _latest_model(folder: Path) -> Path:
    folder = Path(folder)  # <-- key fix
    if not folder.exists():
        raise FileNotFoundError(f"Processed folder not found: {folder}")
    
    files = sorted(folder.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No models in {folder}")
    return files[-1]

def load_data():
    csv_path = _latest_csv(DATA_DIR)
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {csv_path} shape={df.shape}")
    X = df.drop(columns=["churn", "customer_id"], errors="ignore")
    y = df["churn"].astype(int)
    return X, y, csv_path.stem

def evaluate_model(model, X, y, name, version_ts):
    y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }

    # ROC curve if available
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        metrics["roc_auc"] = float(roc_auc)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        plt.plot([0,1],[0,1],"--")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC Curve - {name}")
        roc_path = Path(REPORT_DIR) / f"{name}_roc_{version_ts}.png"
        plt.savefig(roc_path); plt.close()

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    cm_path = Path(REPORT_DIR) / f"{name}_cm_{version_ts}.png"
    plt.savefig(cm_path); plt.close()

    # Classification report
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(6,3))
    sns.heatmap(report_df.iloc[:-1,:-1], annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(f"Classification Report - {name}")
    cr_path = Path(REPORT_DIR) / f"{name}_report_{version_ts}.png"
    plt.savefig(cr_path); plt.close()

    return metrics, report

def main():
    version_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    X, y, dataset_name = load_data()
    model_path = _latest_model(MODEL_DIR)
    model = joblib.load(model_path)
    name = model_path.stem

    metrics, report = evaluate_model(model, X, y, name, version_ts)

    # Save metrics JSON + CSV
    json_path = Path(REPORT_DIR) / f"{name}_metrics_{version_ts}.json"
    with open(json_path, "w") as f: json.dump(metrics, f, indent=2)

    csv_path = Path(REPORT_DIR) / f"{name}_report_{version_ts}.csv"
    pd.DataFrame(report).transpose().to_csv(csv_path)

    print(f"✅ Evaluation complete for {name}")
    print(f"Metrics JSON: {json_path}")
    print(f"Report CSV: {csv_path}")

if __name__ == "__main__":
    main()
