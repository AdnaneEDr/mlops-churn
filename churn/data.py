import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from churn.schema import FEATURES, Customer

COMMIT = "d5371f5d83a446ad5673cbcca3b814b926491f8a"
BASE = f"https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/{COMMIT}"
URL = BASE + "/data/Telco-Customer-Churn.csv"


def acquire(directory=Path("data")):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "telco.csv"
    if not path.exists():
        with urlopen(URL, timeout=60) as response:
            content = response.read()
        path.write_bytes(content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = directory / "provenance.json"
    if manifest.exists() and json.loads(manifest.read_text())["sha256"] != digest:
        raise ValueError("Empreinte des données modifiée : restaurer le fichier original")
    manifest.write_text(
        json.dumps(
            {
                "url": URL,
                "commit": COMMIT,
                "sha256": digest,
                "repository_license": "Apache-2.0",
                "nature": "IBM public sample benchmark; not verified real customers",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    license_path = directory / "IBM-LICENSE.txt"
    if not license_path.exists():
        with urlopen(BASE + "/LICENSE", timeout=60) as response:
            license_path.write_bytes(response.read())
    return path


def validate_frame(raw):
    expected = set(FEATURES) | {"customerID", "Churn"}
    if set(raw.columns) != expected:
        raise ValueError(f"Colonnes incorrectes : attendu {sorted(expected)}")
    if len(raw) < 50 or raw.customerID.isna().any() or raw.customerID.duplicated().any():
        raise ValueError("Taille insuffisante ou identifiants absents/dupliqués")
    if not raw.Churn.isin(["Yes", "No"]).all() or raw.Churn.nunique() != 2:
        raise ValueError("Cible binaire Yes/No obligatoire")
    frame = raw[FEATURES].copy()
    blank = frame.TotalCharges.astype(str).str.strip().eq("")
    frame["TotalCharges"] = pd.to_numeric(frame.TotalCharges.mask(blank), errors="raise")
    for name in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    for row in frame.replace({np.nan: None}).to_dict("records"):
        Customer.model_validate(row)
    quality = {
        "rows": len(raw),
        "missing_total_charges": int(frame.TotalCharges.isna().sum()),
        "churn_rate": float(raw.Churn.eq("Yes").mean()),
    }
    return frame, raw.Churn.eq("Yes").astype(int), quality


if __name__ == "__main__":
    print(acquire())
