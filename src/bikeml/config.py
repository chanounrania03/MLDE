"""Chargement de la configuration.

La configuration vient de configs/config.yaml. Trois surcharges sont possibles
sans toucher au fichier :
  - BIKEML_CONFIG : chemin vers un autre fichier YAML (utile pour les tests) ;
  - MLFLOW_TRACKING_URI : serveur MLflow à utiliser ;
  - BIKEML_OPS_DB : autre base de traces (les tests de charge écrivent dans une
    base séparée pour ne pas mélanger trafic synthétique et trafic réel).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


class Config(dict):
    """Dictionnaire avec accès par attribut et résolution des chemins."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        """Chemin absolu d'une entrée de la section `paths`."""
        if key == "ops_db" and os.environ.get("BIKEML_OPS_DB"):
            p = Path(os.environ["BIKEML_OPS_DB"])
        else:
            p = Path(self["paths"][key])
        return p if p.is_absolute() else self.root / p

    @property
    def root(self) -> Path:
        return Path(self.get("_root", ROOT))

    @property
    def tracking_uri(self) -> str:
        uri = os.environ.get("MLFLOW_TRACKING_URI", self["mlflow"]["tracking_uri"])
        # une URI sqlite relative est rattachée à la racine du projet
        if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
            uri = f"sqlite:///{self.root / uri.removeprefix('sqlite:///')}"
        return uri


@lru_cache(maxsize=4)
def _load(path: str) -> Config:
    with open(path, encoding="utf-8") as fh:
        cfg = Config(yaml.safe_load(fh))
    cfg.setdefault("_root", str(Path(path).resolve().parents[1]))
    return cfg


def get_config() -> Config:
    return _load(os.environ.get("BIKEML_CONFIG", str(DEFAULT_CONFIG)))


def rel(path: Path | str, cfg: Config | None = None) -> str:
    """Chemin affiché relativement à la racine du projet (logs lisibles)."""
    root = (cfg or get_config()).root
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def ensure_dirs(cfg: Config) -> None:
    for key in ("raw_dir", "batches_dir", "incoming_dir", "accepted_dir",
                "quarantine_dir", "reports_dir"):
        cfg.path(key).mkdir(parents=True, exist_ok=True)
    cfg.path("ops_db").parent.mkdir(parents=True, exist_ok=True)
