"""Prévision de la demande de vélos en libre-service : chaîne MLOps complète."""
import os

# Exécution 100 % locale : pas d'envoi de télémétrie par MLflow ni Prefect.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__version__ = "1.0.0"
