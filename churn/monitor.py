import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from churn.schema import CATEGORIES, FEATURES, NUMERIC


def histogram(values, spec):
    series = pd.Series(values)
    if "edges" in spec:
        edges = [-np.inf, *spec["edges"], np.inf]
        counts = np.histogram(series.dropna().astype(float), bins=edges)[0].tolist()
        return counts + [int(series.isna().sum())]
    return [int(series.eq(c).sum()) for c in spec["categories"]] + [
        int((~series.isin(spec["categories"])).sum())
    ]


def reference(frame):
    specs = {}
    for name in FEATURES:
        if name in NUMERIC:
            spec = {
                "edges": np.unique(
                    frame[name].dropna().quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                ).tolist()
            }
        else:
            spec = {"categories": CATEGORIES[name]}
        spec["counts"] = histogram(frame[name], spec)
        specs[name] = spec
    return specs


def psi(expected, observed):
    # Lissage de Laplace identique pour les deux populations.
    p = (np.asarray(expected) + 1) / (sum(expected) + len(expected))
    q = (np.asarray(observed) + 1) / (sum(observed) + len(observed))
    return float(np.sum((q - p) * np.log(q / p)))


def compare_counts(ref, observed, n):
    scores = {name: psi(spec["counts"], observed[name]) for name, spec in ref.items()}
    return {
        "rows": n,
        "minimum_rows": 200,
        "sufficient_sample": n >= 200,
        "warning_threshold": 0.1,
        "alert_threshold": 0.2,
        "psi": scores,
        "alerts": [name for name, score in scores.items() if score >= 0.2] if n >= 200 else [],
        "note": "Dérive des entrées uniquement ; aucune mesure de performance sans étiquettes.",
    }


class Monitor:
    def __init__(self, path, ref, version):
        self.path, self.ref, self.version = str(path), ref, version
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS bins (version TEXT, day TEXT, feature TEXT, "
                "bin INTEGER, count INTEGER, PRIMARY KEY(version,day,feature,bin))"
            )

    def record(self, records, probabilities):
        day = datetime.now(timezone.utc).date().isoformat()
        frame = pd.DataFrame(records)
        counts = {name: histogram(frame[name], spec) for name, spec in self.ref.items()}
        counts["prediction"] = np.histogram(probabilities, bins=np.linspace(0, 1, 11))[0].tolist()
        with sqlite3.connect(self.path, timeout=15) as db:
            for name, values in counts.items():
                db.executemany(
                    "INSERT INTO bins VALUES (?,?,?,?,?) ON CONFLICT "
                    "(version,day,feature,bin) DO UPDATE SET count=count+excluded.count",
                    [(self.version, day, name, i, int(n)) for i, n in enumerate(values)],
                )

    def report(self, day=None):
        day = day or datetime.now(timezone.utc).date().isoformat()
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                "SELECT feature,bin,count FROM bins WHERE version=? AND day=?", (self.version, day)
            ).fetchall()
        observed = {name: [0] * len(spec["counts"]) for name, spec in self.ref.items()}
        prediction = [0] * 10
        for name, i, count in rows:
            if name == "prediction":
                prediction[i] = count
            elif name in observed:
                observed[name][i] = count
        n = sum(prediction)
        report = (
            compare_counts(self.ref, observed, n)
            if n
            else {"rows": 0, "sufficient_sample": False, "alerts": [], "psi": {}}
        )
        return {
            **report,
            "model_version": self.version,
            "day": day,
            "probability_histogram": prediction,
        }


def main():
    from churn.model import load_bundle

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--db", default="monitoring/events.sqlite")
    parser.add_argument("--day")
    args = parser.parse_args()
    _, meta, path = load_bundle()
    ref = json.loads((path / "reference.json").read_text())
    Path("reports").mkdir(exist_ok=True)
    if args.demo:
        frame = pd.read_csv(path / "reference_sample.csv")
        shifted = frame.copy()
        shifted["tenure"] = 1
        shifted["MonthlyCharges"] = 150.0
        shifted["Contract"] = "Month-to-month"
        report = {
            "simulation": True,
            "baseline": compare_counts(
                ref, {k: histogram(frame[k], s) for k, s in ref.items()}, len(frame)
            ),
            "perturbed": compare_counts(
                ref, {k: histogram(shifted[k], s) for k, s in ref.items()}, len(shifted)
            ),
        }
        target = Path("reports/drift_demo.json")
    else:
        report = Monitor(args.db, ref, meta["version"]).report(args.day)
        target = Path("reports/drift_live.json")
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
