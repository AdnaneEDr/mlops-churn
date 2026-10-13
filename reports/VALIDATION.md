# Vérifications effectivement réalisées

Le 13 octobre 2026 sur Windows, Python 3.12, CPU. Aucun GPU ni service payant utilisé.

| Contrôle | Résultat réel |
|---|---|
| Acquisition IBM à une révision fixe | 7 043 lignes téléchargées, licence conservée, SHA-256 enregistré |
| Validation du dataset complet | Réussie ; 11 TotalCharges absents tolérés |
| Entraînement complet | Deux exécutions dans le dépôt, trois candidats chacune |
| MLflow | Runs, artefacts et deux versions du modèle telco-churn enregistrés ; serveur HTTP testé |
| Répétabilité | Même SHA-256 du pipeline et mêmes métriques lors des deux exécutions |
| Modèle sauvegardé et API réelle | Chargement au démarrage, santé, métadonnées, prédictions HTTP vérifiés |
| Individuel / lot | Première prédiction identique, entrée invalide refusée en HTTP 422 |
| Streamlit | Serveur HTTP sain, formulaire testé par AppTest et dans le navigateur |
| CSV de 20 clients | Traitement API, affichage réel après upload et téléchargement navigateur vérifiés |
| Export navigateur | Les 20 résultats téléchargés correspondent exactement au CSV attendu |
| Supervision | Volume et histogrammes persistés ; rapport de dérive live généré |
| Simulation de dérive | 0 alerte sur référence, 3 sur Contract, tenure et MonthlyCharges perturbés |
| pytest | 17 tests réussis, dont entraînement de contrôle synthétique, API et interface |
| Ruff | Lint et format réussis sur les 14 fichiers Python |
| pip check | Aucune dépendance incompatible après résolution et verrouillage |
| Environnement neuf | Venv distinct, installation du lock et package éditable réussies |
| Copie sans artefacts | Acquisition depuis IBM et troisième entraînement complet réussis |
| Modèle issu de la copie neuve | Même empreinte et mêmes métriques ; chargement et API batch via TestClient réussis |
| Tests dans l'environnement neuf | 17 tests réussis, lint et format réussis |
| Docker Compose | Configuration analysée avec succès par docker compose config --quiet |
| Image et démarrage conteneurs | **Non validés** : moteur Docker Linux inaccessible |
| GitHub Actions | Workflow écrit, pas exécuté sur GitHub |
| Cloud | Procédure préparée, aucune ressource créée et aucun déploiement effectué |

Preuves : `experiments.json`, `training_second_run.log`, `live_validation.json`,
`clean_validation.json`, `drift_demo.json`, `drift_live.json`, `browser_predictions.csv`,
`interface_csv.jpg`, `interface_predictions.jpg` et les bundles versionnés dans `models/`.
Les bases et artefacts MLflow locaux restent dans `mlflow.db` et `mlruns/`.

## Limitation Docker et commande à exécuter

Docker CLI et Docker Desktop sont installés. Le lancement de Docker Desktop a été
tenté ; le moteur n'est pas devenu disponible. `docker compose build` et
`docker compose up -d` ont échoué avec :

```text
open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

Aucune construction ni exécution de conteneur réussie n'est revendiquée.
Sur le poste, ouvrir Docker Desktop et attendre que le moteur Linux soit opérationnel,
puis dans le dépôt :

```powershell
docker info
docker compose config --quiet
docker compose build
docker compose up -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
```

Les services Python actuellement lancés occupent 8000, 8501 et 5000 ; les arrêter
avant Compose. Les PID de lancement sont dans `monitoring/processes.json`.
Sous Windows, vérifier aussi le processus propriétaire du port avec
`Get-NetTCPConnection -LocalPort 8000,8501,5000 -State Listen` : le lanceur du venv
peut avoir créé un processus enfant. Arrêter uniquement les processus de ce projet.

## Erreurs corrigées pendant la réalisation

- L'exécutable `python` initial désignait le raccourci Microsoft Store ; utilisation
  explicite de `py -3.12` puis du Python du venv.
- La résolution initiale de SQLAlchemy 2.1 était incompatible avec MLflow 3.5 ;
  version 2.0.44 figée et installation vérifiée par pip check et entraînement.
- Une validation après conversion laissait accepter True comme SeniorCitizen=1 ;
  contrôle du type avant conversion ajouté, avec test de régression.
- Les longueurs et imports détectés par Ruff ont été corrigés avant les contrôles finaux.

MLflow avertit que sa signature contient des entiers : l'API impose que tenure et
SeniorCitizen soient présents et non nuls. Leurs valeurs absentes ne sont donc pas
admises à l'inférence. Les avertissements Streamlit de contexte lors d'AppTest
n'empêchent pas l'exécution ; aucune exception d'interface n'a été observée.

## Reproduire les contrôles

Suivre README.md depuis un clone neuf. Après lancement des trois serveurs :

```powershell
.\.venv\Scripts\python.exe scripts/verify_running.py
.\.venv\Scripts\python.exe -m churn.monitor --demo
.\.venv\Scripts\python.exe -m churn.monitor
.\.venv\Scripts\python.exe scripts/build_report.py
```

Le contrôle CSV headless utilise un point d'injection de test ; l'upload et le
download ont également été réalisés séparément dans le navigateur réel.
Les variables de test ne remplacent pas le modèle de démonstration : celui-ci est
entraîné sur le benchmark IBM téléchargé.
