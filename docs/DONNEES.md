# Provenance et conditions d'utilisation

Source primaire : [dépôt IBM](https://github.com/IBM/telco-customer-churn-on-icp4d),
[README IBM](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/README.md),
[licence du dépôt](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE).
Consultés le 13 octobre 2026. La licence publiée est Apache 2.0. Aucune licence de
données distincte n'a été identifiée dans ce dépôt ; ce constat ne permet pas
d'affirmer des droits supplémentaires sur d'éventuelles données tierces.
L'acquisition utilise le CSV directement publié par IBM avec conservation de sa
licence. Le CSV n'est pas suivi dans Git pour éviter une redistribution inutile.

Révision fixée : `d5371f5d83a446ad5673cbcca3b814b926491f8a`.
URL exacte et SHA-256 : `data/provenance.json`. Une modification locale de la source
est refusée lors de la réacquisition si le manifeste existe.

Le jeu Telco est un **benchmark public d'exemple IBM**. Son téléchargement réel
ne prouve pas que les lignes décrivent une population réelle de clients. Il n'est
pas généré par ce projet. Les seules données synthétiques locales sont les fixtures
de tests ; la perturbation du rapport de dérive est explicitement une simulation.

Validation : colonnes exactes, cible Yes/No avec deux classes, identifiants uniques,
catégories valides, valeurs finies, bornes, cohérence téléphonie/internet.
Une absence de TotalCharges est tolérée et imputée dans la Pipeline.
customerID ne devient ni variable de modèle ni contenu de logs API.
