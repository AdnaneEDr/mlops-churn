import json

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.model import pipeline
from churn.monitor import reference
from churn.schema import CATEGORIES


@pytest.fixture
def customer():
    return {
        **{k: v[0] for k, v in CATEGORIES.items()},
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }


@pytest.fixture
def training_data(customer):
    rows = []
    for i in range(90):
        rows.append(
            {
                **customer,
                "tenure": i % 72,
                "MonthlyCharges": float(20 + i),
                "TotalCharges": None if i % 10 == 0 else float(i * 60),
            }
        )
    return pd.DataFrame(rows), pd.Series([int(i % 3 == 0) for i in range(90)])


@pytest.fixture
def bundle(tmp_path, monkeypatch, training_data):
    import hashlib

    x, y = training_data
    model = pipeline(LogisticRegression(max_iter=1000, random_state=42)).fit(x, y)
    root = tmp_path / "models"
    directory = root / "test-v1"
    directory.mkdir(parents=True)
    joblib.dump(model, directory / "pipeline.joblib")
    meta = {
        "version": "test-v1",
        "threshold": 0.3,
        "selected_model": "logistic",
        "test_metrics": {},
        "pipeline_sha256": hashlib.sha256((directory / "pipeline.joblib").read_bytes()).hexdigest(),
    }
    (directory / "metadata.json").write_text(json.dumps(meta))
    (directory / "reference.json").write_text(json.dumps(reference(x)))
    (root / "current.json").write_text(json.dumps({"version": "test-v1"}))
    monkeypatch.setenv("MODEL_ROOT", str(root))
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    monkeypatch.setenv("MONITOR_DB", str(tmp_path / "events.sqlite"))
    return directory
