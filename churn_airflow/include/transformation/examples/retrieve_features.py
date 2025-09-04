# dags/scripts/examples/retrieve_features.py
import sys
import os
from pathlib import Path

# Add parent folder (scripts/) to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from feature_store import LocalFeatureStore
from pathlib import Path
import pandas as pd
#from feature_store import LocalFeatureStore

def training_retrieval_example():
    fs = LocalFeatureStore()
    # Suppose you have a training list of customer_ids:
    customer_ids = [101, 202, 303, 404]  # example ids

    # Grab a subset of features for model training
    df = fs.get_features(
        name="customer_engagement",
        version="v1",
        entity_ids=customer_ids,
        feature_names=[
            "avg_monthly_spend_per_invoice_scaled",
            "engagement_score_scaled",
            "tenure_proxy_scaled"
        ]
    )
    print("Training features:\n", df.head())

def inference_retrieval_example():
    fs = LocalFeatureStore()
    # For one live (inference) id:
    customer_id = 101
    vec = fs.get_online_vector(
        name="customer_engagement",
        version="v1",
        entity_id=customer_id,
        feature_names=[
            "avg_monthly_spend_per_invoice_scaled",
            "engagement_score_scaled",
            "tenure_proxy_scaled"
        ]
    )
    print(f"Inference vector for {customer_id}:\n", vec)

if __name__ == "__main__":
    training_retrieval_example()
    inference_retrieval_example()
