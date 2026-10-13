import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression
from streamlit.testing.v1 import AppTest

from churn.api import app
from churn.data import validate_frame
from churn.model import load_bundle, pipeline
from churn.monitor import Monitor, compare_counts, histogram, reference
from churn.schema import Customer
from churn.train import choose_threshold, metrics
from churn.ui import batch_from_csv, csv_records


def test_quality_and_imputation(training_data):
    x, y = training_data
    raw = x.copy()
    raw["customerID"] = [f"test-{i}" for i in range(len(x))]
    raw["Churn"] = y.map({0: "No", 1: "Yes"})
    raw["TotalCharges"] = raw.TotalCharges.fillna(" ")
    cleaned, target, quality = validate_frame(raw)
    assert quality["missing_total_charges"] == 9
    train = cleaned.iloc[:60]
    model = pipeline(LogisticRegression(max_iter=1000)).fit(train, target.iloc[:60])
    median = (
        model.named_steps["prepare"]
        .named_transformers_["numeric"]
        .named_steps["impute"]
        .statistics_[-1]
    )
    assert median == train.TotalCharges.median()
    assert np.isfinite(model.predict_proba(cleaned.iloc[60:])).all()
    assert target.equals(y)
    raw.loc[1, "customerID"] = raw.loc[0, "customerID"]
    with pytest.raises(ValueError):
        validate_frame(raw)


@pytest.mark.parametrize(
    "change",
    [
        {"tenure": "12"},
        {"MonthlyCharges": -1.0},
        {"gender": "other"},
        {"unexpected": 1},
        {"tenure": 1.5},
        {"MonthlyCharges": float("inf")},
        {"PhoneService": "No"},
        {"SeniorCitizen": True},
    ],
)
def test_strict_schema(customer, change):
    with pytest.raises(ValidationError):
        Customer.model_validate({**customer, **change})


def test_bundle_integrity(bundle, customer):
    model, meta, _ = load_bundle()
    assert meta["version"] == "test-v1"
    assert model.predict_proba(pd.DataFrame([customer])).shape == (1, 2)
    with (bundle / "pipeline.joblib").open("ab") as file:
        file.write(b"corruption")
    with pytest.raises(ValueError, match="corrompu"):
        load_bundle()


def test_api_consistency_and_invalid(bundle, customer):
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"]
        assert client.get("/model").json()["version"] == "test-v1"
        one = client.post("/predict", json=customer).json()
        batch = client.post("/predict/batch", json={"customers": [customer, customer]}).json()
        assert batch["predictions"] == [one, one]
        assert 0 <= one["churn_probability"] <= 1
        for payload in [{}, {**customer, "tenure": "12"}, {**customer, "email": "secret"}]:
            response = client.post("/predict", json=payload)
            assert response.status_code == 422
            assert "secret" not in response.text
        assert client.post("/predict/batch", json={"customers": []}).status_code == 422
        report = client.get("/monitoring").json()
        assert report["rows"] == 3
        assert not report["sufficient_sample"]


def test_monitor_aggregation_and_drift(tmp_path, training_data):
    x, _ = training_data
    x = pd.concat([x] * 4, ignore_index=True)
    ref = reference(x)
    monitor = Monitor(tmp_path / "monitor.sqlite", ref, "v1")
    monitor.record(x.to_dict("records"), [0.3] * len(x))
    assert monitor.report()["rows"] == len(x)
    assert max(monitor.report()["psi"].values()) < 1e-9
    shifted = x.copy()
    shifted["tenure"] = 1
    result = compare_counts(ref, {k: histogram(shifted[k], s) for k, s in ref.items()}, len(x))
    assert "tenure" in result["alerts"]
    other = Monitor(tmp_path / "monitor.sqlite", ref, "v2")
    assert other.report()["rows"] == 0


def test_business_threshold():
    y = [0, 0, 1, 1]
    prob = np.array([0.1, 0.4, 0.3, 0.9])
    threshold = choose_threshold(y, prob)
    assert (
        metrics(y, prob, threshold)["cost_per_customer"]
        < metrics(y, prob, 0.5)["cost_per_customer"]
    )


def test_csv_validation(customer):
    content = pd.DataFrame([customer]).to_csv(index=False).encode()
    assert csv_records(content)[0]["tenure"] == 12
    with pytest.raises(ValueError):
        csv_records(b"email\nprivate@example.com\n")
    with pytest.raises(ValueError):
        csv_records(pd.DataFrame([customer] * 1001).to_csv(index=False).encode())


def test_ui_form_and_csv_through_api(bundle, customer):
    with TestClient(app) as client:

        def request(path, payload=None):
            result = client.get(path) if payload is None else client.post(path, json=payload)
            result.raise_for_status()
            return result.json()

        content = pd.DataFrame([customer, customer]).to_csv(index=False).encode()
        with patch("churn.ui.api_request", side_effect=request):
            assert len(batch_from_csv(content)) == 2
            at = AppTest.from_file("app.py", default_timeout=30)
            at.run()
            assert not at.exception
            at.button[0].click().run()
            assert not at.exception
            assert len(at.metric) == 1
            at.session_state["_test_csv_bytes"] = content
            at.run()
            assert not at.exception
            assert len(at.dataframe) == 1
            assert len(at.dataframe[0].value) == 2


def test_missing_model_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError), TestClient(app):
        pass


def test_schema_saved(bundle):
    # Le fixture simule un petit entraînement de contrôle sans téléchargement.
    metadata = json.loads((bundle / "metadata.json").read_text())
    assert len(metadata["pipeline_sha256"]) == 64
