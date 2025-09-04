# dags/scripts/feature_store.py
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yaml
import os

FEATURE_STORE_ROOT = Path("/usr/local/airflow/dags/feature_store")
REGISTRY_PATH = FEATURE_STORE_ROOT / "registry.yaml"

#BASE_DIR = Path(__file__).resolve().parent.parent.parent  # points to c:\churn_airflow
#FEATURE_STORE_ROOT = BASE_DIR / "include" / "data" / "feature_store"
#REGISTRY_PATH = FEATURE_STORE_ROOT / "registry.yaml"

log_file = Path(os.environ.get("FEATURE_STORE_LOG_FILE", "/usr/local/airflow/dags/data/logs/feature_store.log"))
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

class LocalFeatureStore:
    """
    A simple filesystem-based feature store:
      - Metadata registry (YAML)
      - Feature sets as Parquet files
      - Versioned by `name` + `version` (e.g., customer_engagement v1)
      - Entity key: customer_id
    """

    def __init__(self, root: Path = FEATURE_STORE_ROOT, registry_path: Path = REGISTRY_PATH):
        self.root = Path(root)
        self.registry_path = Path(registry_path)
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_registry({"feature_sets": {}})

    # ---------- Registry helpers ----------
    def _read_registry(self) -> Dict:
        with open(self.registry_path, "r") as f:
            return yaml.safe_load(f) or {"feature_sets": {}}

    def _write_registry(self, data: Dict):
        with open(self.registry_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def register_feature_set(
        self,
        name: str,
        version: str,
        description: str,
        source: str,
        primary_key: str,
        features: Dict[str, Dict[str, str]],
        materialization_path: str,
    ):
        """
        Register/overwrite a feature set in the registry.

        features: {
          "avg_monthly_spend_per_invoice": {"dtype": "float", "description": "..."},
          "engagement_score": {"dtype": "float", "description": "..."},
          ...
        }
        """
        reg = self._read_registry()
        if "feature_sets" not in reg:
            reg["feature_sets"] = {}

        reg["feature_sets"][f"{name}:{version}"] = {
            "name": name,
            "version": version,
            "description": description,
            "source": source,
            "primary_key": primary_key,
            "features": features,
            "materialization_path": str(materialization_path),
        }
        self._write_registry(reg)
        logging.info(f"Registered feature set {name}:{version} -> {materialization_path}")

    def get_registry(self) -> Dict:
        return self._read_registry()

    # ---------- Materialization ----------
    def materialize_from_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        version: str,
        primary_key: str,
        keep_columns: List[str],
        description: str,
        source: str,
    ) -> Path:
        """
        Save the selected features as a Parquet file and register them.
        """
        # Keep only [primary_key] + requested features
        cols = [primary_key] + [c for c in keep_columns if c in df.columns]
        feat_df = df[cols].drop_duplicates(subset=[primary_key])

        # Save parquet
        out_path = self.root / f"{name}_{version}.parquet"
        feat_df.to_parquet(out_path, index=False)

        # Prepare registry metadata
        features_meta = {
            c: {
                "dtype": str(feat_df[c].dtype),
                "description": self._auto_description(c)
            } for c in keep_columns if c in feat_df.columns
        }

        self.register_feature_set(
            name=name,
            version=version,
            description=description,
            source=source,
            primary_key=primary_key,
            features=features_meta,
            materialization_path=str(out_path),
        )
        return out_path

    def _auto_description(self, col: str) -> str:
        # quick defaults; you can hand-edit in registry.yaml later
        mapping = {
            "avg_monthly_spend_per_invoice": "Monthly spend normalized by invoice count.",
            "engagement_score": "Composite engagement metric from sessions, pages and time on site.",
            "tenure_proxy": "Proxy for tenure: last_seen_at - sign_up_date.",
        }
        return mapping.get(col, f"Feature `{col}`.")

    # ---------- Retrieval ----------
    def get_features(
        self,
        name: str,
        version: str,
        entity_ids: List,
        feature_names: Optional[List[str]] = None,
        primary_key_override: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve a feature DataFrame for the given entity_ids.
        """
        reg = self._read_registry()
        key = f"{name}:{version}"
        if key not in reg.get("feature_sets", {}):
            raise ValueError(f"Feature set {key} not found in registry")

        info = reg["feature_sets"][key]
        parquet_path = Path(info["materialization_path"])
        pk = primary_key_override or info["primary_key"]

        feat_df = pd.read_parquet(parquet_path)

        # filter by entity_ids
        subset = feat_df[feat_df[pk].isin(entity_ids)].copy()

        if feature_names:
            cols = [pk] + [c for c in feature_names if c in subset.columns]
            subset = subset[cols]

        return subset

    def get_online_vector(
        self,
        name: str,
        version: str,
        entity_id,
        feature_names: Optional[List[str]] = None,
        primary_key_override: Optional[str] = None
    ) -> Dict:
        """
        Retrieve features as a dict for a single entity (inference-style).
        """
        df = self.get_features(
            name=name,
            version=version,
            entity_ids=[entity_id],
            feature_names=feature_names,
            primary_key_override=primary_key_override
        )
        if df.empty:
            return {}
        row = df.iloc[0].to_dict()
        # Pop the primary key from vector (optional)
        pk = primary_key_override or self._read_registry()["feature_sets"][f"{name}:{version}"]["primary_key"]
        row.pop(pk, None)
        return row
