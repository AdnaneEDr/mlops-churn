# Expliquer le projet en entretien

Un modèle utile en notebook devient un système exploitable lorsqu'on sait reproduire
ses données, son entraînement et son comportement à l'inférence. Ici, le bundle
contient le prétraitement et le classifieur : l'API n'a pas à réimplémenter l'encodage.

1. **Split avant apprentissage.** Les statistiques d'imputation, de standardisation
   et les catégories observées sont apprises sur train. La CV encapsule la Pipeline
   pour que ces statistiques ne voient jamais le pli évalué.
2. **Trois niveaux de comparaison.** Dummy vérifie que les variables apportent un
   signal. La logistique est rapide et relativement explicable. Le boosting capte
   les relations non linéaires. On conserve le gagnant mesuré ; on ne suppose pas
   qu'un modèle complexe sera meilleur.
3. **Probabilité et décision.** Le seuil 0,5 ne convient pas nécessairement à une
   campagne de rétention. Un faux négatif coûte ici cinq fois un faux positif.
   Ce coût n'est pas estimé dans les données : c'est une hypothèse à discuter.
4. **Test réservé.** La validation sert aux choix ; le test décrit ensuite la qualité
   du système choisi. Consulter le test pour ajuster les paramètres invaliderait
   cette séparation. Les trois candidats n'ont pas de métriques test séparées.
5. **MLflow et bundle.** MLflow garde la trace des expériences et des versions de
   registre. Le bundle local permet une inférence autonome ; sa version et son
   empreinte relient la prédiction à l'expérience et aux données.
6. **Contrat d'entrée.** Une catégorie mal orthographiée ou un entier sous forme
   de texte est refusé en JSON. Un fichier CSV passe par une conversion tabulaire
   explicite puis la même validation. Les erreurs ne contiennent pas les valeurs.
7. **Supervision sans labels.** Les histogrammes donnent volume et changement de
   distribution. Le PSI n'explique pas la cause et ne mesure ni rappel ni ROC-AUC.
   Pour mesurer la performance après déploiement, il faudrait joindre les scores
   aux labels de churn observés ultérieurement, avec une politique de conservation.
8. **Déploiement local.** Compose organise API, interface et MLflow. Les données
   persistantes sont montées à l'extérieur de l'image ; une reconstruction ne les
   efface pas. La santé API conditionne le lancement de l'interface.

Limites : dataset d'exemple, pas de validation temporelle/externe, coûts artificiels,
probabilités non calibrées, biais potentiels des variables démographiques, métriques
sans intervalles de confiance, API locale sans authentification, suivi limité à un
processus/SQLite. Pas de causalité : le score n'établit pas qu'une offre empêchera le churn.
Suite raisonnable : validation temporelle réelle, calibration sur partition séparée,
audit par groupes, mesure de capacité de campagne, collecte différée des labels,
authentification/TLS et sauvegardes automatiques.
