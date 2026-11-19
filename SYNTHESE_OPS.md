# Synthèse des choix et résultats OPS

Projet : prévision de la demande horaire de vélos en libre-service (UCI Bike Sharing).
Les chiffres ci-dessous proviennent de l'exécution complète enregistrée dans `notebooks/projet_mlops_velos.ipynb` (Python 3.13, machine de 4 vCPU et 16 Go de RAM). Ils se reproduisent avec les commandes du README.

---

## 1. Décisions techniques

| Décision | Choix | Raison |
|---|---|---|
| Orchestrateur | Prefect 3 | flows en Python, retries et hooks d'échec déclaratifs, exécution locale sans serveur, UI optionnelle ; Airflow jugé trop lourd pour un poste local |
| Suivi et registre | MLflow, base SQLite locale | suivi des runs, artefacts et registre ; les **alias** (`@champion`, `@previous`) découplent l'API des numéros de version et rendent le rollback immédiat |
| Service | FastAPI + uvicorn | validation stricte par Pydantic, OpenAPI automatique, multi-workers pour le dimensionnement |
| Traces | SQLite en mode WAL | aucun service à installer, requêtable en SQL, lisible par le dashboard pendant que l'API écrit |
| Supervision | Streamlit + `/metrics` Prometheus | dashboard léger branché sur la base ; `/metrics` reste compatible Grafana |
| Charge | Locust headless + psutil | scénario en Python, export CSV exploité automatiquement, mesure CPU/RAM côté serveur |
| Modèle | HistGradientBoosting, cible log(1+cnt), pondération par l'ancienneté (demi-vie 30 j) | rapide, gère les interactions heure × jour ouvré ; la pondération et l'exclusion de `mnth`/`season` répondent à la forte croissance de la demande 2011 → 2012 (RMSE moyen sur les 12 évaluations mensuelles : 72,8 contre 119,4 avec toutes les variables et 130,5 pour la référence naïve) |
| Évaluation | découpage temporel (28 derniers jours) | imite l'usage réel ; un découpage aléatoire serait trop optimiste |
| Mise en service | 3 règles : RMSE < 200 ; meilleur que la référence naïve ; ≤ champion × 1,02 **sur la même période** | les erreurs de mois différents ne sont pas comparables (saisonnalité) |
| Déploiement | Docker : une image, 4 services (MLflow, pipeline, API, dashboard) via `docker-compose.yml`, volumes pour les données et le registre | reproductible sur n'importe quel poste ; MLflow 3 lancé avec `--allowed-hosts` pour accepter les appels des autres conteneurs |
| Échec de dépendance | API démarrée en mode dégradé (503 explicite), délais HTTP bornés, ancien modèle conservé si un rechargement échoue | un registre en panne ne doit pas rendre l'API indisponible ni la bloquer au démarrage |
| Rejeu des données | lot initial 2011 + un lot par mois de 2012 | permet de démontrer l'arrivée de nouvelles données, le réentraînement et les décisions de promotion |

## 2. Preuves d'intégration (exécution de bout en bout)

**Nominal**
- 13 lots ingérés (8 645 lignes pour 2011, puis 692 à 744 lignes par mois) : tous validés, acceptés, et chacun a déclenché un entraînement tracé dans MLflow (paramètres, métriques, empreinte des données, manifeste des lots, commit, signature du modèle).
- 13 versions enregistrées, 13 promues selon les règles ; la dernière (v13, RMSE 80,6) l'a été grâce à la tolérance de 2 % face à un champion à 80,2.
- Déclenchement automatique démontré : le lot de décembre a été détecté et traité par `python -m bikeml watch` sans intervention.
- Une version volontairement sous-entraînée (`retrain --set max_iter=30`, RMSE 85,9) a été **rejetée** (85,9 > 80,6 × 1,02) : tag `rejected`, alerte WARNING, champion inchangé.
- API : prédictions unitaire et par lot servies par v13 ; 300 feedbacks réels rattachés aux prédictions → erreur mesurée en production pour v13 : **MAE 40,6, RMSE 65,9** ; un second feedback pour la même prédiction est refusé (409).

**Incident n°1 — lot invalide** (`release-bad` : colonne manquante, valeurs manquantes, heures à 27, comptages négatifs, doublons)
- 5 contrôles en échec sur 8, chacun identifiant son défaut ; arrêt avant entraînement ; lot en `data/quarantine/` ; rapport JSON ; alertes `validation` et `prefect` ; run `REJECTED` ; code retour du pipeline = 1 ; aucune nouvelle version ; champion v13 inchangé.

**Incident n°2 — registre MLflow indisponible**
- Instance d'API démarrée avec un serveur MLflow éteint : démarrage en quelques secondes en état `degraded`, cause exposée dans `/health`, `/predict` → **503 « modèle indisponible »**, alerte `api` enregistrée.
- Serveur MLflow redémarré → `POST /admin/reload` → v13 chargée, service nominal **sans redémarrage**.

**Retour arrière**
- `rollback` : champion v13 → v12 (alias `@previous`), API notifiée et rechargée à chaud, prédictions suivantes servies par v12 ; `@previous` passe sur v11 (jamais sur la version retirée, pour qu'un second rollback ne la remette pas en service) ; puis `rollback --version 13` pour revenir à l'état nominal. Chaque bascule est ajoutée à la justification d'origine (tags `status`, `decision`).

**Tests automatisés** : 33 tests (validation dont valeurs non numériques, modèle, règles de promotion, API dont feedback en double et jeton d'administration, chaîne complète avec registre temporaire, rollback simple et double), tous au vert, sans réseau.

## 3. Tests de charge

**Protocole.** Scénario Locust : 10 prédictions unitaires : 2 lots de 24 : 1 feedback : 1 santé ; attente de 50 à 200 ms entre requêtes. Configurations : 1 et 2 workers uvicorn, chacun limité à un thread de calcul (`OMP_NUM_THREADS=1`). Paliers : 10, 25 et 50 utilisateurs, 20 s chacun. Mesures : débit, p50/p95/p99, erreurs (Locust) ; CPU et RAM de l'API et de ses workers (psutil, chaque seconde). L'API testée écrit dans une base de traces séparée (`data/ops_loadtest.db`). SLO : **p95 ≤ 200 ms, erreurs ≤ 1 %**.

| Workers | Utilisateurs | Débit (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Erreurs | CPU moyen | RAM max |
|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 53,1 | 50 | 160 | 210 | 0 % | 77 % | 384 Mo |
| 1 | 25 | 56,3 | 330 | 520 | 600 | 0 % | 98 % | 391 Mo |
| 1 | 50 | 55,6 | 750 | 1000 | 1200 | 0 % | 98 % | 398 Mo |
| 2 | 10 | 63,9 | 23 | 71 | 120 | 0 % | 78 % | 803 Mo |
| 2 | 25 | 93,6 | 110 | 360 | 530 | 0 % | 138 % | 811 Mo |
| 2 | 50 | 91,9 | 390 | 860 | 1100 | 0 % | 141 % | 823 Mo |

(CPU en % d'un cœur : 100 % = un cœur plein. API et générateur de charge sur la même machine : chiffres pessimistes. Une campagne antérieure sur une machine de 2 vCPU plus rapide par cœur donnait environ deux fois plus de débit : les valeurs absolues sont à refaire sur la machine cible.)

**Lecture.**
- Le service est **limité par le CPU** : un worker plafonne vers 56 req/s (un cœur saturé) ; au-delà, les requêtes attendent en file et la latence monte, **sans erreur**.
- Le 2e worker porte le débit maximal de 56 à 94 req/s (**× 1,66**) et ramène la p95 de 160 à 71 ms à 10 utilisateurs.
- La mémoire croît d'environ **400 Mo par worker** ; elle n'est jamais le facteur limitant.
- Optimisation appliquée grâce aux mesures : `OMP_NUM_THREADS=1` par worker. Sans elle, chaque prédiction lançait un pool de threads sur tous les cœurs (1 worker, 10 utilisateurs : CPU 138 % → 76 %, p50 120 → 57 ms).

## 4. Dimensionnement recommandé

Demande : ≈ 6,4 req/s par utilisateur actif (attente moyenne de 125 ms). Capacité : ≈ 50 req/s par worker (1 vCPU) sous le SLO, soit **≈ 35 req/s par worker avec 30 % de marge** → *workers = débit cible ÷ 35*.

| Charge visée | Recommandation | Justification |
|---|---|---|
| ≈ 10 utilisateurs (≈ 64 req/s) | **2 workers, 2 vCPU, 1 Go** | p95 mesurée 71 ms ; 1 worker tient le SLO de justesse (p95 160 ms) mais sature déjà |
| ≈ 25 utilisateurs (≈ 160 req/s) | **≈ 5 workers / 5 vCPU, ~2 Go, sur 2 instances** | extrapolation : aucune config. testée ne tient le SLO (2 workers : p95 360 ms) |
| ≈ 50 utilisateurs (≈ 320 req/s) | **≈ 9-10 workers : 3 instances de 3-4 vCPU, ~4 Go** | extrapolation (2 workers : p95 860 ms) |

Les extrapolations sont à valider sur l'infrastructure cible avec `python scripts/run_load_tests.py --workers 1 2 4`.

**Optimisations restantes identifiées par les mesures** : écriture des traces SQLite asynchrone et groupée ; conversion directe des entrées en tableau NumPy au lieu d'un DataFrame pandas pour les prédictions unitaires.

## 5. Limites connues

- SQLite convient à une instance locale ; en multi-instance, il faudrait une base serveur (PostgreSQL) et une file de messages pour les traces.
- Compteurs Prometheus par processus : avec plusieurs workers, `/metrics` ne reflète que le worker qui répond (mode multiprocess de `prometheus_client` non activé). Le dashboard s'appuie sur SQLite, qui agrège tous les workers.
- L'arrivée des données et les feedbacks sont simulés à partir de l'historique.
- Pas de détection de dérive des entrées (seule la qualité via feedback est suivie).
- `/admin/reload` n'est protégé que par un jeton partagé optionnel (`BIKEML_ADMIN_TOKEN`) ; en production il faudrait une vraie authentification et HTTPS.
- Mesures de charge réalisées sur un poste de développement.
- Pas d'intégration continue (piste : GitHub Actions lançant `pytest` et la construction de l'image Docker).
