"""Chargement et détention du modèle servi.

Le service charge la version portant l'alias `champion` dans le registre.
Comportement en cas de dépendance indisponible (registre MLflow KO) :
  - au démarrage : l'API démarre quand même, en état « dégradé » ; /predict
    répond 503 avec un message clair au lieu de planter ;
  - lors d'un rechargement : l'ancienne version reste en service et l'erreur
    est remontée (alerte + /health).
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bikeml.config import Config, get_config

log = logging.getLogger("bikeml.api.model")

# Si le registre est un serveur MLflow distant injoignable, on veut échouer
# vite (quelques secondes) plutôt que bloquer le démarrage plusieurs minutes.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")


class ModelUnavailable(Exception):
    pass


class ModelService:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or get_config()
        self.model = None
        self.version: str | None = None
        self.loaded_at: str | None = None
        self.last_error: str | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load_champion(self) -> bool:
        from bikeml.model import registry  # import tardif : mlflow est lourd

        alias = self.cfg.promotion["champion_alias"]
        try:
            version = registry.get_alias_version(alias, self.cfg)
            if version is None:
                raise ModelUnavailable(f"aucune version avec l'alias '{alias}'")
            if version == self.version:
                return True
            model = registry.load_version(version, self.cfg)
            # préchauffage : le premier appel à predict est plus lent
            model.predict(pd.DataFrame([dict.fromkeys(self.cfg.model["features"], 0)]))
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:300]
            log.error("Chargement du champion impossible : %s", self.last_error)
            return False
        with self._lock:
            self.model, self.version = model, version
            self.loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.last_error = None
        log.info("Modèle v%s chargé", version)
        return True

    def set_model(self, model, version: str) -> None:
        """Injection directe (tests unitaires)."""
        self.model, self.version = model, version
        self.loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def predict(self, rows: list[dict]) -> tuple[list[float], str]:
        with self._lock:
            model, version = self.model, self.version
        if model is None:
            raise ModelUnavailable(self.last_error or "aucun modèle chargé")
        X = pd.DataFrame(rows)[self.cfg.model["features"]]
        preds = np.clip(model.predict(X), 0, None)
        return [round(float(p), 2) for p in preds], version
