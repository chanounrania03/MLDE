# Raccourcis. Toutes les commandes passent par `python -m bikeml` (voir src/bikeml/__main__.py).
PY ?= python

.PHONY: install setup pipeline release release-bad watch api dashboard mlflow-ui test load-test rollback models reset

install:        ## installe les dépendances figées + le paquet en mode éditable
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

setup:          ## télécharge les données et prépare les lots de rejeu
	$(PY) -m bikeml setup

release:        ## dépose le prochain lot historique dans data/incoming
	$(PY) -m bikeml release

release-bad:    ## dépose un lot corrompu (incident contrôlé)
	$(PY) -m bikeml release-bad

pipeline:       ## exécute le pipeline sur les lots en attente
	$(PY) -m bikeml run

watch:          ## déclenchement automatique à l'arrivée d'un lot
	$(PY) -m bikeml watch

api:            ## lance l'API sur http://127.0.0.1:8000 (docs : /docs)
	$(PY) -m bikeml api

dashboard:      ## lance le dashboard de supervision sur http://localhost:8501
	streamlit run dashboard/app.py

mlflow-ui:      ## interface MLflow sur http://127.0.0.1:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

models:         ## liste les versions du registre
	$(PY) -m bikeml models

rollback:       ## remet en service la version précédente
	$(PY) -m bikeml rollback

test:           ## tests automatisés
	$(PY) -m pytest

load-test:      ## campagne de tests de charge + recommandation
	$(PY) scripts/run_load_tests.py

reset:          ## repart de zéro (supprime données générées, registre et traces)
	rm -rf data mlruns mlflow.db reports/validation reports/load_tests
