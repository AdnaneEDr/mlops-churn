"""Réévaluation explicite du même test réservé, sans sélection ni réentraînement."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from churn.data import validate_frame
from churn.model import load_bundle
from churn.train import metrics

if __name__ == "__main__":
    model, metadata, path = load_bundle()
    if hashlib.sha256(Path("data/telco.csv").read_bytes()).hexdigest() != metadata["data_sha256"]:
        raise ValueError("Les données diffèrent de celles de l'entraînement")
    x, y, _ = validate_frame(pd.read_csv("data/telco.csv"))
    indices = json.loads((path / "split_indices.json").read_text())["test"]
    measured = metrics(
        y.loc[indices], model.predict_proba(x.loc[indices])[:, 1], metadata["threshold"]
    )
    print(json.dumps(measured, indent=2))
