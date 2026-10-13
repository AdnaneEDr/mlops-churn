"""Générer le rapport à partir des mesures enregistrées, sans nouvel entraînement."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    reports = Path("reports")
    results = json.loads((reports / "experiments.json").read_text())
    eda = json.loads((reports / "eda.json").read_text())
    drift = json.loads((reports / "drift_demo.json").read_text())
    meta = results["selected"]
    lines = [
        "# Expériences et résultats mesurés",
        "",
        "Exécution du 13 octobre 2026, CPU, "
        "Python 3.12. Les valeurs suivantes sont extraites des JSON produits par le pipeline.",
        "",
        "## Données et exploration",
        "",
        f"{eda['rows']:,} lignes ; churn : {eda['churn_rate']:.2%} ; "
        f"{eda['missing_total_charges']} TotalCharges absents.",
        "",
        f"Split stratifié : {meta['split_sizes']['train']} train, "
        f"{meta['split_sizes']['validation']} validation, {meta['split_sizes']['test']} test.",
        "",
        "Taux de churn descriptif selon le contrat (benchmark complet) :",
        "",
    ]
    for contract, rate in eda["churn_by_contract"].items():
        lines.append(f"- {contract} : {rate:.2%}")
    lines.extend(
        [
            "",
            "Cette association ne démontre pas de causalité. Ces statistiques "
            "descriptives ne servent pas à apprendre le prétraitement.",
            "",
            "## Comparaison sur validation",
            "",
            "CV 3 plis sur train, scoring AP ; 3 C logistiques et 8 configurations "
            "de boosting. Aucun choix guidé par le test.",
            "",
            "| Candidat | AP CV | AP validation | ROC-AUC | Seuil | Rappel | Coût/client |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["candidates"]:
        m = row["validation"]
        cv = f"{row['cv_pr_auc']:.4f}" if row.get("cv_pr_auc") is not None else "—"
        lines.append(
            f"| {row['model']} | {cv} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | "
            f"{row['threshold']:.2f} | {m['recall']:.2%} | {m['cost_per_customer']:.4f} |"
        )
    lines.extend(["", "Paramètres choisis par CV :", ""])
    for row in results["candidates"]:
        lines.append(f"- {row['model']} : `{json.dumps(row.get('best_parameters', {}))}`")
    lines.extend(
        [
            "",
            f"Gagnant : **{meta['selected_model']}**, seuil **{meta['threshold']:.2f}**.",
            "Le boosting obtient une AP de validation légèrement supérieure, mais la "
            "logistique minimise le coût FN=5, FP=1. Le choix métier prévaut sur cette "
            "petite différence de classement. Pas de gain économique réel revendiqué.",
            "",
            "## Test réservé du gagnant",
            "",
            "| Mesure | Valeur |",
            "|---|---:|",
        ]
    )
    measured = meta["test_metrics"]
    for name in ["precision", "recall", "f1", "roc_auc", "pr_auc", "cost_per_customer"]:
        lines.append(f"| {name} | {measured[name]:.6f} |")
    lines.extend(
        [
            "",
            "AP = average_precision_score ; pas l'aire trapézoïdale de la courbe PR.",
            "",
            "| Réel / prédit | Pas de churn | Churn |",
            "|---|---:|---:|",
            f"| Pas de churn | {measured['tn']} | {measured['fp']} |",
            f"| Churn | {measured['fn']} | {measured['tp']} |",
            "",
            "![Matrice de confusion](confusion_matrix.png)",
            "",
            f"{measured['tp']} clients en churn détectés, {measured['fn']} manqués et "
            f"{measured['fp']} faux positifs. Une équipe de rétention doit vérifier sa "
            "capacité avant d'adopter ce seuil. La probabilité n'est pas garantie calibrée.",
            "",
            "## Version et reproductibilité",
            "",
            f"Bundle : `{meta['version']}` ; registre MLflow : `{meta['registry_version']}`.",
            f"Données SHA-256 : `{meta['data_sha256']}`.",
            f"Pipeline SHA-256 : `{meta['pipeline_sha256']}`.",
            "",
            "Deux entraînements complets avec la graine 42 ont été exécutés ; les deux "
            "bundles conservés ont la même empreinte de pipeline et les mêmes métriques "
            "test. Cela démontre la répétabilité dans cet environnement, sans garantir "
            "l'identité bit à bit sur un autre système ou une autre bibliothèque BLAS.",
            "",
            "## Simulation de dérive",
            "",
            f"{drift['baseline']['rows']} observations de référence : "
            f"{len(drift['baseline']['alerts'])} alerte. Observations perturbées : "
            f"{len(drift['perturbed']['alerts'])} alertes.",
            "",
        ]
    )
    for name in drift["perturbed"]["alerts"]:
        lines.append(f"- {name} : PSI={drift['perturbed']['psi'][name]:.4f}")
    lines.extend(
        [
            "",
            "Simulation explicite, seuil d'alerte 0,20, minimum 200 observations. "
            "Aucune performance de production mesurée sans labels.",
            "",
            "## Limites",
            "",
            "Données d'exemple IBM, partition aléatoire sans validation temporelle, "
            "coûts hypothétiques, absence de calibration et d'intervalles de confiance, "
            "risques de biais démographiques et généralisation non démontrée. "
            "Validation finale technique : [VALIDATION.md](VALIDATION.md).",
        ]
    )
    (reports / "RESULTATS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    contracts = eda["churn_by_contract"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(contracts), [100 * v for v in contracts.values()], color="#2563eb")
    ax.set(ylabel="Taux de churn (%)", title="IBM Telco : churn par contrat (descriptif)")
    fig.tight_layout()
    fig.savefig(reports / "churn_by_contract.png", dpi=160)
    plt.close(fig)
    cv = [
        "# Trois propositions pour le CV",
        "",
        "- Développement et exécution d'un pipeline MLOps de churn sur 7 043 observations "
        "IBM Telco : validation, prétraitement sans fuite, comparaison de trois modèles "
        "et suivi des expériences avec registre MLflow.",
        f"- Déploiement local Python d'une API FastAPI et d'une interface Streamlit "
        f"pour prédictions individuelles et CSV ; ROC-AUC test {measured['roc_auc']:.3f} "
        f"et rappel {measured['recall']:.1%} au seuil {meta['threshold']:.2f}, "
        "avec validation stricte et export des résultats.",
        "- Mise en place de 17 tests automatisés, contrôles Ruff, logs structurés et "
        "détection de dérive par PSI ; préparation de Docker Compose, GitHub Actions "
        "et d'une procédure de livraison avec retour arrière.",
        "",
        "Ne pas ajouter « déployé dans le cloud », « Docker validé » "
        "ou « CI exécutée sur GitHub » : "
        "ces actions n'ont pas été réalisées. Les chiffres concernent le benchmark réservé, "
        "pas une mesure de production.",
    ]
    Path("docs/CV.md").write_text("\n".join(cv) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
