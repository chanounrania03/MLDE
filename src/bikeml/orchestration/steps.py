"""Étapes du pipeline, écrites comme de simples fonctions Python.

Elles ne dépendent pas de Prefect : on peut les tester unitairement et les
appeler à la main. Le fichier flows.py se contente de les enchaîner en tâches
Prefect (retries, journalisation, états, hooks d'échec).
"""
from __future__ import annotations

import logging
import os
import shutil
import urllib.request
import uuid
from pathlib import Path

import pandas as pd

from bikeml.config import Config, ensure_dirs, get_config
from bikeml.data.validate import DataValidationError, validate_batch
from bikeml.model import registry
from bikeml.model.train import save_json, train_and_register
from bikeml.monitoring.alerts import send_alert
from bikeml.monitoring.store import OpsStore

log = logging.getLogger("bikeml.steps")


def list_incoming(cfg: Config | None = None) -> list[Path]:
    cfg = cfg or get_config()
    ensure_dirs(cfg)
    return sorted(cfg.path("incoming_dir").glob("*.csv"))


def validate_step(batch: Path, run_id: str, cfg: Config) -> int:
    """Contrôle qualité bloquant. Un lot non conforme part en quarantaine."""
    store = OpsStore(cfg.path("ops_db"))
    store.update_run(run_id, step="validation")
    try:
        df = pd.read_csv(batch)
    except Exception as exc:  # fichier illisible = lot non conforme
        df = pd.DataFrame()
        log.error("Lecture impossible de %s : %s", batch.name, exc)
    report = validate_batch(df, cfg)
    save_json({"batch": batch.name, **report.to_dict()},
              cfg.path("reports_dir") / "validation" / f"{batch.stem}.json")
    if not report.passed:
        dest = cfg.path("quarantine_dir") / batch.name
        shutil.move(str(batch), dest)
        err = DataValidationError(report)
        store.update_run(run_id, status="REJECTED", n_rows=report.n_rows, message=str(err))
        send_alert(f"Lot {batch.name} rejeté et mis en quarantaine. {err}",
                   source="validation")
        raise err
    store.update_run(run_id, n_rows=report.n_rows)
    return report.n_rows


def accept_step(batch: Path, cfg: Config) -> Path:
    dest = cfg.path("accepted_dir") / batch.name
    shutil.move(str(batch), dest)
    return dest


def train_step(batch_name: str, run_id: str, cfg: Config):
    OpsStore(cfg.path("ops_db")).update_run(run_id, step="training")
    return train_and_register(cfg, trigger_batch=batch_name, pipeline_run_id=run_id)


def promotion_step(result, run_id: str, cfg: Config) -> dict:
    store = OpsStore(cfg.path("ops_db"))
    store.update_run(run_id, step="promotion")
    decision, chall, champ = registry.evaluate_and_decide(
        result.version, result.holdout, cfg, baseline_rmse=result.baseline["rmse"])
    if decision.promote:
        registry.promote(result.version, decision.reason, cfg)
    else:
        registry.reject(result.version, decision.reason, cfg)
        send_alert(f"Version {result.version} non promue : {decision.reason}",
                   level="WARNING", source="registry")
    summary = {"version": result.version, "promoted": decision.promote,
               "reason": decision.reason, "challenger": chall, "champion": champ,
               "baseline": result.baseline}
    store.update_run(run_id, model_version=result.version, promoted=int(decision.promote),
                     rmse=chall["rmse"], message=decision.reason)
    return summary


def manual_retrain(params_override: dict | None = None, cfg: Config | None = None) -> dict:
    """Réentraînement à la demande (nouveaux hyperparamètres, par exemple).

    Il passe par exactement les mêmes règles de mise en service que le pipeline.
    """
    cfg = cfg or get_config()
    store = OpsStore(cfg.path("ops_db"))
    run_id = uuid.uuid4().hex[:12]
    store.start_run(run_id, "reentrainement-manuel")
    try:
        result = train_and_register(cfg, "reentrainement-manuel", run_id, params_override)
        store.update_run(run_id, step="promotion", n_rows=len(result.holdout))
        summary = promotion_step(result, run_id, cfg)
        if summary["promoted"]:
            notify_api(cfg)
        store.update_run(run_id, status="SUCCESS", step="done")
        return summary
    except Exception as exc:
        store.update_run(run_id, status="FAILED", message=str(exc)[:500])
        send_alert(f"Réentraînement manuel en échec : {exc}", source="pipeline")
        raise


def notify_api(cfg: Config) -> None:
    """Demande à l'API (si elle tourne) de recharger le champion. Jamais bloquant."""
    url = os.environ.get("BIKEML_API_URL",
                         f"http://{cfg.api['host']}:{cfg.api['port']}") + "/admin/reload"
    headers = {}
    if os.environ.get("BIKEML_ADMIN_TOKEN"):
        headers["X-Admin-Token"] = os.environ["BIKEML_ADMIN_TOKEN"]
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="POST", headers=headers),
                               timeout=3)
        log.info("API notifiée : rechargement du champion.")
    except Exception:
        log.info("API non joignable : elle chargera le champion à son prochain démarrage.")
