#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
import logging
import json
import os

# ---- Setup ----
MODELS_DIR = "/usr/local/airflow/dags/data/models"

#BASE_DIR = Path(__file__).resolve().parent.parent
#MODELS_DIR = BASE_DIR / "data" / "models"

log_file = Path(os.environ.get("DEPLOY_LOG_FILE", "/usr/local/airflow/dags/data/logs/deployment.log"))
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

app = FastAPI(title="Churn Prediction API", version="1.0")

# ---- Load Latest Model + Features ----
# ---- Load Latest Model + Features ----
def load_latest_model():
    model_files = sorted(Path(MODELS_DIR).glob("*.pkl"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not model_files:
        raise FileNotFoundError(f"No models found in {MODELS_DIR}")

    latest_model = model_files[0]

    # get full timestamp from model name (last 2 parts)
    parts = latest_model.stem.split("_")
    ts = "_".join(parts[-2:])   # e.g. 20250824_173522

    feature_file = Path(MODELS_DIR) / f"features_{ts}.json"
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature list not found: {feature_file}")

    with open(feature_file, "r") as f:
        feature_list = json.load(f)

    logging.info(f"Loading model -> {latest_model}")
    return joblib.load(latest_model), latest_model.name, feature_list

model, model_name, feature_list = load_latest_model()

# ---- Request Schema ----
class CustomerFeatures(BaseModel):
    features: dict

# ---- Endpoints ----
@app.get("/")
def root():
    return {
        "status": "ok",
        "model": model_name,
        "expected_features": feature_list
    }

@app.post("/predict")
def predict(data: CustomerFeatures):
    req_df = pd.DataFrame([data.features])

    # align to training features (drop unknowns, add missing as 0)
    df = req_df.reindex(columns=feature_list, fill_value=0)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1] if hasattr(model, "predict_proba") else None

    return {
        "model": model_name,
        "prediction": int(pred),
        "probability": float(prob) if prob is not None else None,
        "missing_features": [c for c in feature_list if c not in req_df.columns],
        "extra_features_ignored": [c for c in req_df.columns if c not in feature_list]
    }
