# Livraison et retour arrière

Une release associe commit Git, lock des dépendances, image immuable et bundle modèle.
Le bundle versionné inclut pipeline, métadonnées, schéma, référence et indices de split.
La version MLflow est distincte du nom de bundle et enregistrée dans metadata.json.

```bash
docker build -t mlops-churn:1.0.0 .
docker image inspect mlops-churn:1.0.0 --format '{{.Id}}'
docker save -o mlops-churn-1.0.0.tar mlops-churn:1.0.0
```

Avant livraison : lint, format, tests, entraînement explicite si nouveau modèle,
rapport test, intégrité bundle et démonstration de dérive. Une release change de tag
(ex. 1.0.1), sans écraser l'image précédente. Copie des bundles dans `models/` ;
conserver au moins la précédente version. Configurer `.env` :

```dotenv
IMAGE_TAG=1.0.0
MODEL_VERSION=<nom-exact-du-dossier-version>
```

```bash
docker compose up -d --no-build api ui
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/model
```

Envoyer aussi un client connu et comparer sa probabilité à la release validée.
Rollback : remettre les anciens IMAGE_TAG et MODEL_VERSION, puis exécuter les trois
commandes précédentes. L'API recharge le bundle seulement au démarrage.
Chaque version conserve ses propres histogrammes ; ne pas fusionner les références.
Ce déploiement redémarre le service et peut provoquer une brève indisponibilité.

Sauvegarder les bundles et les dossiers MLflow. Pour une base SQLite utilisée,
utiliser l'API `sqlite3.Connection.backup` ou arrêter les services avant copie ;
copier uniquement le fichier .sqlite pendant des écritures WAL n'est pas fiable.
Restaurer et contrôler les empreintes avant lancement. Ne pas utiliser
`docker compose down -v` dans une procédure de mise à jour.
Le workflow CI construit l'image ; il n'effectue ni publication ni déploiement.
