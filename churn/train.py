import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from churn.data import acquire, validate_frame
from churn.model import pipeline
from churn.monitor import reference
from churn.schema import Customer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42


def metrics(y, probability, threshold):
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "cost_per_customer": float((5 * fn + fp) / len(y)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold(y, probability):
    values = [
        (metrics(y, probability, float(t))["cost_per_customer"], float(t))
        for t in np.linspace(0.01, 0.99, 99)
    ]
    return min(values)[1]


@threadpool_limits.wrap(limits=2)
def train(data_path=None, smoke=False):
    path = Path(data_path) if data_path else acquire()
    raw = pd.read_csv(path)
    x, y, quality = validate_frame(raw)
    if smoke:
        indices, _ = train_test_split(x.index, train_size=400, stratify=y, random_state=SEED)
        x, y = x.loc[indices], y.loc[indices]
    train_x, rest_x, train_y, rest_y = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=SEED
    )
    valid_x, test_x, valid_y, test_y = train_test_split(
        rest_x, rest_y, test_size=0.5, stratify=rest_y, random_state=SEED
    )
    reports = Path("reports/smoke" if smoke else "reports")
    reports.mkdir(parents=True, exist_ok=True)
    summary = {
        **quality,
        "training_rows_used": len(x),
        "splits": {"train": len(train_x), "validation": len(valid_x), "test": len(test_x)},
        "numeric_summary": x.describe().to_dict(),
        "churn_by_contract": raw.groupby("Contract")
        .Churn.apply(lambda s: float(s.eq("Yes").mean()))
        .to_dict(),
    }
    (reports / "eda.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + str(Path("mlflow.db").resolve()))
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("churn-smoke" if smoke else "churn-portfolio")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    candidates = [
        ("dummy", DummyClassifier(strategy="prior"), {}),
        (
            "logistic",
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"classifier__C": [0.1, 1, 10]},
        ),
        (
            "gradient_boosting",
            HistGradientBoostingClassifier(random_state=SEED, max_iter=100),
            {
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_leaf_nodes": [7, 15],
                "classifier__l2_regularization": [1.0, 10.0],
            },
        ),
    ]
    results, fitted = [], {}
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, estimator, grid in candidates:
        model = pipeline(estimator)
        with mlflow.start_run(run_name=name) as run:
            parameters, cv_score = {}, None
            if grid:
                if smoke:
                    grid = {k: v[:1] for k, v in grid.items()}
                search = GridSearchCV(model, grid, scoring="average_precision", cv=cv, n_jobs=1)
                search.fit(train_x, train_y)
                model = search.best_estimator_
                parameters, cv_score = search.best_params_, float(search.best_score_)
                mlflow.log_params(search.best_params_)
                mlflow.log_metric("cv_pr_auc", search.best_score_)
            else:
                model.fit(train_x, train_y)
            prob = model.predict_proba(valid_x)[:, 1]
            threshold = choose_threshold(valid_y, prob)
            measured = metrics(valid_y, prob, threshold)
            mlflow.log_params(
                {
                    "seed": SEED,
                    "threshold": threshold,
                    "fn_cost": 5,
                    "fp_cost": 1,
                    "data_sha256": fingerprint,
                    "smoke": smoke,
                }
            )
            mlflow.log_metrics({"validation_" + k: v for k, v in measured.items()})
            mlflow.log_artifact(str(reports / "eda.json"))
            results.append(
                {
                    "model": name,
                    "run_id": run.info.run_id,
                    "threshold": threshold,
                    "validation": measured,
                    "best_parameters": parameters,
                    "cv_pr_auc": cv_score,
                }
            )
            fitted[name] = model
    winner = min(
        results, key=lambda r: (r["validation"]["cost_per_customer"], -r["validation"]["pr_auc"])
    )
    model = fitted[winner["model"]]
    test_probability = model.predict_proba(test_x)[:, 1]
    final = metrics(test_y, test_probability, winner["threshold"])
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + winner["run_id"][:8]
    if smoke:
        version = "smoke-" + version
    directory = Path("models") / version
    directory.mkdir(parents=True)
    joblib.dump(model, directory / "pipeline.joblib")
    metadata = {
        "version": version,
        "selected_model": winner["model"],
        "threshold": winner["threshold"],
        "test_metrics": final,
        "data_sha256": fingerprint,
        "seed": SEED,
        "smoke": smoke,
        "split_sizes": summary["splits"],
        "run_id": winner["run_id"],
        "best_parameters": winner["best_parameters"],
        "pipeline_sha256": hashlib.sha256((directory / "pipeline.joblib").read_bytes()).hexdigest(),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (directory / "schema.json").write_text(
        json.dumps(Customer.model_json_schema(), indent=2), encoding="utf-8"
    )
    (directory / "reference.json").write_text(
        json.dumps(reference(train_x), indent=2), encoding="utf-8"
    )
    train_x.sample(min(500, len(train_x)), random_state=SEED).to_csv(
        directory / "reference_sample.csv", index=False
    )
    split_ids = {
        "train": train_x.index.tolist(),
        "validation": valid_x.index.tolist(),
        "test": test_x.index.tolist(),
    }
    (directory / "split_indices.json").write_text(json.dumps(split_ids), encoding="utf-8")
    test_x.head(20).to_csv(reports / "demo_customers.csv", index=False)
    ConfusionMatrixDisplay.from_predictions(test_y, test_probability >= winner["threshold"])
    plt.title("Jeu de test réservé")
    plt.savefig(reports / "confusion_matrix.png", bbox_inches="tight")
    plt.close()
    result = {
        "candidates": results,
        "selected": metadata,
        "business_assumption": "Coût relatif FN=5, FP=1 ; hypothèse pédagogique, pas un ROI mesuré",
        "pr_auc_definition": "average_precision_score (AP), pas intégration trapézoïdale",
    }
    (reports / "experiments.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with mlflow.start_run(run_id=winner["run_id"]):
        mlflow.log_metrics({"test_" + k: v for k, v in final.items()})
        info = mlflow.sklearn.log_model(
            model,
            name="pipeline",
            registered_model_name="churn-smoke" if smoke else "telco-churn",
            signature=infer_signature(train_x.head(5), model.predict(train_x.head(5))),
            input_example=train_x.head(5),
        )
        metadata["mlflow_model_uri"] = info.model_uri
        metadata["registry_version"] = str(info.registered_model_version)
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (reports / "experiments.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        for artifact in directory.glob("*.json"):
            mlflow.log_artifact(str(artifact), artifact_path="bundle")
        mlflow.log_artifacts(str(reports), artifact_path="reports")
    pointer = Path("models/smoke-current.json" if smoke else "models/current.json")
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(json.dumps({"version": version}), encoding="utf-8")
    temporary.replace(pointer)
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train(args.data, args.smoke)
