"""Préparation des variables, entraînement et suivi MLflow.

Choix du modèle : un HistGradientBoostingRegressor de scikit-learn.
  - il gère nativement les variables catégorielles et les interactions
    (heure x jour ouvré, très marquées dans ce jeu) ;
  - il s'entraîne en quelques secondes sur CPU ;
  - la cible est transformée en log(1 + cnt) car les comptages sont très
    asymétriques ; la prédiction est ramenée dans l'échelle d'origine.

Variables écartées volontairement :
  - casual et registered : leur somme EST la cible (fuite de données) ;
  - yr : en production on ne verrait que des années « nouvelles » ;
  - mnth, season : la demande a fortement augmenté entre 2011 et 2012 ; ces
    variables rattachent la prédiction au niveau du même mois l'an passé et
    font sous-estimer la demande (constaté par l'expérience du notebook) ;
  - instant, dteday : identifiants temporels.

Pour suivre l'évolution de la demande, les observations sont pondérées selon
leur ancienneté (demi-vie configurable) : le modèle colle aux semaines récentes.

Évaluation : découpage temporel (les N derniers jours disponibles), jamais
aléatoire, pour imiter l'usage réel (prédire le futur à partir du passé).
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from bikeml.config import Config, get_config

log = logging.getLogger("bikeml.train")


@dataclass
class TrainResult:
    run_id: str
    version: str
    metrics: dict
    baseline: dict
    holdout: pd.DataFrame


class NaiveBaseline:
    """Référence naïve : moyenne historique par (heure, jour ouvré).

    Un modèle de ML n'a d'intérêt que s'il fait mieux que cette règle simple ;
    c'est l'une des conditions de mise en service.
    """

    keys = ["hr", "workingday"]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveBaseline":
        self.table_ = y.groupby([X[k] for k in self.keys]).mean()
        self.default_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(X[self.keys])
        return self.table_.reindex(idx).fillna(self.default_).to_numpy()


def load_accepted_data(cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    files = sorted(cfg.path("accepted_dir").glob("*.csv"))
    if not files:
        raise RuntimeError("Aucun lot validé : rien à entraîner.")
    df = pd.concat([pd.read_csv(f, parse_dates=["dteday"]) for f in files], ignore_index=True)
    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    return df, [f.name for f in files]


def data_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def temporal_split(df: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["dteday"].max() - pd.Timedelta(days=holdout_days - 1)
    return df[df["dteday"] < cutoff], df[df["dteday"] >= cutoff]


def recency_weights(df: pd.DataFrame, half_life_days: float | None) -> np.ndarray | None:
    """Poids décroissants avec l'ancienneté : 1 pour le dernier jour, 0,5 à une demi-vie..."""
    if not half_life_days:
        return None
    age = (df["dteday"].max() - df["dteday"]).dt.days.to_numpy()
    return 0.5 ** (age / half_life_days)


def build_model(cfg: Config, params: dict | None = None) -> TransformedTargetRegressor:
    m = cfg.model
    cat_mask = [f in m["categorical"] for f in m["features"]]
    reg = HistGradientBoostingRegressor(categorical_features=cat_mask,
                                        random_state=cfg.project["random_seed"],
                                        **(params or m["params"]))
    return TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1)


def evaluate(model, df: pd.DataFrame, cfg: Config) -> dict:
    X, y = df[cfg.model["features"]], df[cfg.model["target"]]
    pred = np.clip(model.predict(X), 0, None)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


# Sérialisation sûre du modèle. Les versions récentes de MLflow enregistrent les
# modèles scikit-learn avec skops, qui refuse par défaut tout type « non déclaré »
# (protection contre les fichiers malveillants). Les types internes varient selon
# la version de scikit-learn : on les détecte donc sur le modèle lui-même, et on
# n'accepte que ceux de scikit-learn / NumPy (plus functools.partial). Tout autre
# type bloque l'enregistrement.
ALLOWED_TYPE_PREFIXES = ("sklearn.", "numpy.", "functools.partial")


def trusted_types(model) -> list[str]:
    import skops.io as sio

    found = sio.get_untrusted_types(data=sio.dumps(model))
    unexpected = [t for t in found if not t.startswith(ALLOWED_TYPE_PREFIXES)]
    if unexpected:
        raise RuntimeError(f"Types non autorisés dans le modèle : {unexpected}")
    return found


def _serialization_kwargs(model) -> dict:
    import inspect
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    if "skops_trusted_types" not in params:  # anciennes versions de MLflow (pickle)
        return {}
    return {"skops_trusted_types": trusted_types(model)}


def setup_mlflow(cfg: Config) -> None:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.mlflow["experiment"])


def train_and_register(cfg: Config | None = None, trigger_batch: str = "manual",
                       pipeline_run_id: str | None = None,
                       params_override: dict | None = None) -> TrainResult:
    cfg = cfg or get_config()
    params = {**cfg.model["params"], **(params_override or {})}
    setup_mlflow(cfg)
    df, files = load_accepted_data(cfg)
    train_df, holdout = temporal_split(df, cfg.model["holdout_days"])
    feats, target = cfg.model["features"], cfg.model["target"]

    with mlflow.start_run(run_name=f"train-{trigger_batch}") as run:
        model = build_model(cfg, params)
        model.fit(train_df[feats], train_df[target],
                  sample_weight=recency_weights(train_df, cfg.model.get("recency_half_life_days")))
        metrics = evaluate(model, holdout, cfg)
        baseline = evaluate(NaiveBaseline().fit(train_df[feats], train_df[target]), holdout, cfg)
        train_metrics = {f"train_{k}": v for k, v in evaluate(model, train_df, cfg).items()}

        # --- traçabilité des données
        mlflow.log_params({
            "model_type": "HistGradientBoostingRegressor+log1p",
            "recency_half_life_days": cfg.model.get("recency_half_life_days"),
            **{f"hp_{k}": v for k, v in params.items()},
            "features": ",".join(feats),
            "trigger_batch": trigger_batch,
            "n_batches": len(files),
            "n_train": len(train_df),
            "n_holdout": len(holdout),
            "train_start": str(train_df["dteday"].min().date()),
            "train_end": str(train_df["dteday"].max().date()),
            "holdout_start": str(holdout["dteday"].min().date()),
            "holdout_end": str(holdout["dteday"].max().date()),
        })
        mlflow.set_tags({
            "data_sha256": data_fingerprint(df),
            "git_commit": _git_commit(),
            "pipeline_run_id": pipeline_run_id or "manual",
        })
        mlflow.log_metrics({**metrics, **train_metrics,
                            **{f"baseline_{k}": v for k, v in baseline.items()}})
        mlflow.log_dict({"batches": files, "rows": len(df)}, "data_manifest.json")

        # --- artefact modèle + enregistrement dans le registre
        sample = train_df[feats].head(5)
        signature = infer_signature(sample, model.predict(sample))
        info = mlflow.sklearn.log_model(model, name="model", signature=signature,
                                        input_example=sample,
                                        registered_model_name=cfg.model["name"],
                                        **_serialization_kwargs(model))
        version = str(info.registered_model_version)
        log.info("Modèle v%s entraîné : RMSE=%.1f MAE=%.1f R2=%.3f (référence naïve RMSE=%.1f)",
                 version, metrics["rmse"], metrics["mae"], metrics["r2"], baseline["rmse"])

    return TrainResult(run.info.run_id, version, metrics, baseline, holdout)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
