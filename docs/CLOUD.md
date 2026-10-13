# Option cloud : VM AWS EC2 et stockage EBS

**Configuration préparée ; aucune ressource créée, aucun déploiement cloud réalisé.**
Cette option peut être payante et doit être autorisée séparément. Une VM Ubuntu
avec 2–4 vCPU et 8–16 Go de RAM suffit pour cette démonstration CPU.
Docker Compose réutilise les services locaux. Pas de Fargate avec disque éphémère
pour les bases SQLite de ce projet.

Documentation officielle vérifiée le 13 octobre 2026 :
- [EBS volumes](https://docs.aws.amazon.com/ebs/latest/userguide/ebs-volumes.html)
- [Montage EBS Linux](https://docs.aws.amazon.com/ebs/latest/userguide/ebs-using-volumes.html)
- [Connexion EC2 SSH](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-to-linux-instance.html)
- [Docker sur Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Ordre de démarrage Compose](https://docs.docker.com/compose/how-tos/startup-order/)

Après autorisation, créer la VM et un volume EBS chiffré, avec suppression à la
terminaison désactivée pour le volume de données. Groupe de sécurité : SSH uniquement
depuis l'adresse du propriétaire. Ne pas ouvrir 5000/8000/8501 publiquement.
Configurer un budget et des snapshots EBS selon les besoins.

1. Installer Docker Engine et son plugin Compose selon la documentation Ubuntu.
2. Identifier le volume via `lsblk`, vérifier son contenu ; **ne pas formater un
   volume contenant des données**. Monter le volume à `/srv/churn` selon le guide EBS,
   puis ajouter l'UUID à `/etc/fstab` avec `nofail`.
3. Transférer le projet, modèle et lock vers `/srv/churn` avec SCP, ou transférer
   l'image obtenue par `docker save` puis `docker load -i mlops-churn-1.0.0.tar`.
4. Appliquer la configuration de [LIVRAISON.md](LIVRAISON.md) et exécuter :

```bash
cd /srv/churn
mkdir -p models monitoring tracking data reports
sudo chown -R 10001:10001 models monitoring tracking data reports
cp .env.example .env
# Éditer IMAGE_TAG et MODEL_VERSION dans .env
docker compose up -d --no-build
curl --fail http://127.0.0.1:8000/health
docker compose ps
```

5. Depuis le poste utilisateur, établir des tunnels (remplacer IP et clé) :

```bash
ssh -i key.pem -L 8501:127.0.0.1:8501 -L 8000:127.0.0.1:8000 -L 5000:127.0.0.1:5000 ubuntu@IP
```

Ouvrir localhost:8501. Les ports restent privés et les données sur EBS survivent aux
reconstructions d'image. Le service mlflow de Compose possède son propre registre ;
effectuer l'entraînement avec le profil `training` pour y enregistrer le modèle.
Pour accès multiutilisateur : ajouter un reverse proxy TLS et une authentification,
avant toute exposition réseau. SQLite reste adapté à une petite VM unique ; pour
plusieurs replicas, migrer le backend MLflow vers PostgreSQL et les artefacts vers S3.
Les snapshots et une restauration testée restent nécessaires : persistance ≠ sauvegarde.
