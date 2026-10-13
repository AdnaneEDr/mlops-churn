"""Contrôle HTTP et Streamlit headless avec le modèle réellement servi."""

import json
from pathlib import Path

import httpx
import pandas as pd
from streamlit.testing.v1 import AppTest

from churn.ui import batch_from_csv, csv_records


def main():
    base = "http://127.0.0.1:8000"
    content = Path("reports/demo_customers.csv").read_bytes()
    records = csv_records(content)
    with httpx.Client(timeout=30) as client:
        assert client.get(base + "/health").json()["status"] == "ok"
        metadata = client.get(base + "/model").json()
        one = client.post(base + "/predict", json=records[0])
        one.raise_for_status()
        batch = client.post(base + "/predict/batch", json={"customers": records})
        batch.raise_for_status()
        assert one.json() == batch.json()["predictions"][0]
        invalid = client.post(base + "/predict", json={**records[0], "tenure": "bad"})
        assert invalid.status_code == 422
        assert client.get("http://127.0.0.1:8501/_stcore/health").status_code == 200
        assert client.get("http://127.0.0.1:5000/health").status_code == 200
    output = batch_from_csv(content)
    exported = output.to_csv(index=False)
    Path("reports/demo_predictions.csv").write_text(exported, encoding="utf-8")
    assert len(pd.read_csv("reports/demo_predictions.csv")) == len(records)
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception and len(at.metric) == 1
    at.session_state["_test_csv_bytes"] = content
    at.run()
    assert not at.exception and len(at.dataframe[0].value) == len(records)
    report = {
        "model_version": metadata["version"],
        "api_health": True,
        "streamlit_health": True,
        "mlflow_health": True,
        "individual_batch_equal": True,
        "invalid_input_http_status": invalid.status_code,
        "csv_rows": len(records),
        "csv_export_roundtrip": True,
        "streamlit_form_headless": True,
        "streamlit_csv_headless": True,
        "file_upload_check": "AppTest injecte les octets dans le même chemin de traitement ; "
        "pas de sélection manuelle via le dialogue du navigateur",
        "example_prediction": one.json(),
    }
    Path("reports/live_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
