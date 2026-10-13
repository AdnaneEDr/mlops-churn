# Démonstration de cinq minutes

Prérequis : entraînement terminé, API et interface lancées. Préparer les fichiers
`reports/demo_customers.csv`, `reports/experiments.json`, `reports/drift_demo.json`.

- **0:00–0:45** Présenter le besoin : prioriser des clients à contacter. Montrer le
  diagramme du README, la provenance et la distinction entre benchmark et clientèle réelle.
- **0:45–1:30** Montrer les runs MLflow et le rapport réel : baseline, modèles,
  séparation 60/20/20, coût FN/FP, seuil et résultat test du gagnant.
- **1:30–2:30** Dans Streamlit, saisir un client puis prédire. Lire probabilité,
  classe, seuil et version. Provoquer une incohérence (PhoneService=No avec
  MultipleLines=Yes) pour montrer l'erreur explicite.
- **2:30–3:15** Importer `reports/demo_customers.csv`, montrer l'ordre des résultats
  et télécharger `predictions.csv`. Expliquer que Streamlit appelle FastAPI.
- **3:15–4:15** Ouvrir `/monitoring`, puis le rapport de simulation : histogrammes,
  variables perturbées et alertes PSI. Expliquer qu'il n'y a pas de mesure de
  performance sans labels post-prédiction.
- **4:15–5:00** Montrer tests, CI et procédure de rollback. Dire précisément quelles
  validations ont été exécutées localement selon `reports/VALIDATION.md`, puis
  expliquer les limites et le statut du cloud (aucune ressource déployée).
