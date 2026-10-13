import hashlib
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.schema import CATEGORIES, FEATURES, NUMERIC


def pipeline(estimator):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORIES),
            ),
        ]
    )
    return Pipeline([("prepare", preprocessor), ("classifier", estimator)])


def resolve_model():
    root = Path(os.getenv("MODEL_ROOT", "models"))
    version = (
        os.getenv("MODEL_VERSION") or json.loads((root / "current.json").read_text())["version"]
    )
    if Path(version).name != version or version in (".", ".."):
        raise ValueError("Version invalide")
    return root / version


def load_bundle(path=None):
    path = Path(path) if path else resolve_model()
    metadata = json.loads((path / "metadata.json").read_text())
    content = (path / "pipeline.joblib").read_bytes()
    if hashlib.sha256(content).hexdigest() != metadata["pipeline_sha256"]:
        raise ValueError("Pipeline corrompu : empreinte incorrecte")
    # Ne charger que des artefacts de confiance : joblib peut exécuter du code.
    return joblib.load(path / "pipeline.joblib"), metadata, path


def predict(model, metadata, records):
    frame = pd.DataFrame(records)[FEATURES]
    probabilities = model.predict_proba(frame)[:, 1]
    return [
        {
            "churn_probability": float(p),
            "predicted_class": int(p >= metadata["threshold"]),
            "threshold": metadata["threshold"],
            "model_version": metadata["version"],
        }
        for p in probabilities
    ]
