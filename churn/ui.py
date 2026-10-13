import os
from io import BytesIO

import httpx
import numpy as np
import pandas as pd
import streamlit as st

from churn.schema import CATEGORIES, FEATURES, Customer


def api_request(path, payload=None):
    url = os.getenv("API_URL", "http://127.0.0.1:8000") + path
    response = (
        httpx.get(url, timeout=30) if payload is None else httpx.post(url, json=payload, timeout=30)
    )
    if response.status_code == 422:
        raise ValueError(f"Entrées invalides : {response.json()['detail']}")
    response.raise_for_status()
    return response.json()


def csv_records(content):
    if len(content) > 2_000_000:
        raise ValueError("CSV limité à 2 Mo")
    frame = pd.read_csv(BytesIO(content))
    if set(frame.columns) != set(FEATURES):
        raise ValueError("Le CSV doit contenir exactement les 19 variables du modèle")
    if not 1 <= len(frame) <= 1000:
        raise ValueError("Le CSV doit contenir entre 1 et 1000 clients")
    records = frame.replace({np.nan: None}).to_dict("records")
    return [Customer.model_validate(row).model_dump() for row in records]


def batch_from_csv(content):
    records = csv_records(content)
    output = api_request("/predict/batch", {"customers": records})
    return pd.DataFrame(output["predictions"])


def main():
    st.set_page_config(page_title="Churn · MLOps", layout="wide")
    st.title("MLOps — Prédiction du churn client")
    st.caption(
        "Démonstration pédagogique • IBM Telco • décisions de rétention à valider humainement"
    )
    try:
        metadata = api_request("/model")
        st.caption(f"Modèle {metadata['selected_model']} · version {metadata['version']}")
    except (httpx.HTTPError, ValueError) as exc:
        st.error(f"API indisponible : {exc}")
        st.stop()
    one, batch, performance = st.tabs(["Un client", "CSV par lot", "Performances et limites"])
    with one:
        with st.form("customer"):
            columns = st.columns(3)
            values = {}
            for i, (name, choices) in enumerate(CATEGORIES.items()):
                values[name] = columns[i % 3].selectbox(name, choices)
            values["SeniorCitizen"] = columns[0].selectbox("SeniorCitizen", [0, 1])
            values["tenure"] = columns[1].number_input("tenure (mois)", 0, 120, 12)
            values["MonthlyCharges"] = columns[2].number_input("MonthlyCharges", 0.0, 500.0, 70.0)
            missing_total = columns[0].checkbox("TotalCharges inconnu")
            total = columns[1].number_input("TotalCharges", 0.0, 100000.0, 840.0)
            values["TotalCharges"] = None if missing_total else total
            submitted = st.form_submit_button("Prédire")
        if submitted:
            try:
                result = api_request("/predict", Customer.model_validate(values).model_dump())
                st.metric("Probabilité de churn", f"{result['churn_probability']:.1%}")
                st.json(result)
            except (ValueError, httpx.HTTPError) as exc:
                st.error(str(exc))
    with batch:
        st.write("CSV avec les 19 variables, sans customerID ni Churn ; maximum 1 000 lignes.")
        st.download_button(
            "Télécharger un exemple CSV",
            pd.DataFrame([values]).to_csv(index=False),
            "clients_exemple.csv",
            "text/csv",
        )
        upload = st.file_uploader("Importer des clients", type="csv")
        # Point d'injection utilisé par AppTest : même chemin que le fichier uploadé.
        test_content = st.session_state.get("_test_csv_bytes")
        if upload is not None or test_content is not None:
            try:
                content = upload.getvalue() if upload is not None else test_content
                output = batch_from_csv(content)
                st.dataframe(output, width="stretch")
                st.download_button(
                    "Exporter les prédictions",
                    output.to_csv(index=False),
                    "predictions.csv",
                    "text/csv",
                )
            except (ValueError, httpx.HTTPError) as exc:
                st.error(str(exc))
    with performance:
        st.json(metadata["test_metrics"])
        st.write(
            f"Seuil fixé sur validation : {metadata['threshold']:.2f}. Coûts relatifs : "
            "faux négatif = 5, faux positif = 1. Ces coûts sont une hypothèse métier."
        )
        st.warning(
            "Benchmark d'exemple IBM, pas une validation sur votre clientèle. "
            "Probabilités non garanties calibrées. Données historiques, biais possibles "
            "(genre, âge), absence d'étude causale. Sans labels de production, "
            "la supervision mesure la dérive des entrées, pas la performance."
        )
        if st.button("Afficher la supervision"):
            try:
                st.json(api_request("/monitoring"))
            except (httpx.HTTPError, ValueError) as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
