"""Orchestration Prefect du pipeline.

Enchaînement pour chaque lot déposé dans data/incoming :

    validation  ->  acceptation  ->  entraînement  ->  évaluation/promotion  ->  notification API
       |                                  |
       | lot non conforme                 | erreur technique : 2 nouvelles tentatives
       v                                  v
    quarantaine + alerte              échec du run + alerte (hook on_failure)

Déclenchement automatique, deux possibilités :
  - `python -m bikeml watch` : boucle légère qui surveille data/incoming et
    lance le flow dès qu'un fichier arrive (aucun serveur nécessaire) ;
  - `python -m bikeml serve` : déploiement Prefect planifié (toutes les N
    secondes), visible dans l'interface Prefect (`prefect server start`).
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

# Pas de télémétrie Prefect (et évite des verrous SQLite inutiles en mode local).
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from bikeml.config import get_config, rel
from bikeml.data.validate import DataValidationError
from bikeml.monitoring.alerts import send_alert
from bikeml.monitoring.store import OpsStore
from bikeml.orchestration import steps

log = logging.getLogger("bikeml.flows")
_ORCH = get_config().orchestration
RETRY = {"retries": _ORCH["retries"], "retry_delay_seconds": _ORCH["retry_delay_seconds"]}


@task(name="validation", cache_policy=NO_CACHE)
def validate_task(batch: str, run_id: str) -> int:
    # pas de retry : un contrôle qualité est déterministe
    return steps.validate_step(Path(batch), run_id, get_config())


@task(name="acceptation", cache_policy=NO_CACHE)
def accept_task(batch: str) -> str:
    return str(steps.accept_step(Path(batch), get_config()))


@task(name="entrainement", cache_policy=NO_CACHE, **RETRY)
def train_task(batch_name: str, run_id: str):
    return steps.train_step(batch_name, run_id, get_config())


@task(name="promotion", cache_policy=NO_CACHE, **RETRY)
def promotion_task(result, run_id: str) -> dict:
    return steps.promotion_step(result, run_id, get_config())


@task(name="notification-api", cache_policy=NO_CACHE)
def notify_task() -> None:
    steps.notify_api(get_config())


def _on_failure(flow, flow_run, state):
    send_alert(f"Le flow {flow_run.name} a échoué : {state.message}", source="prefect")


@flow(name="traiter-un-lot", on_failure=[_on_failure])
def process_batch(batch: str) -> dict:
    logger = get_run_logger()
    cfg = get_config()
    store = OpsStore(cfg.path("ops_db"))
    run_id = uuid.uuid4().hex[:12]
    name = Path(batch).name
    store.start_run(run_id, name)
    logger.info("Run %s : traitement du lot %s", run_id, name)
    try:
        validate_task(batch, run_id)
        accept_task(batch)
        result = train_task(name, run_id)
        summary = promotion_task(result, run_id)
        notify_task()
        store.update_run(run_id, status="SUCCESS", step="done")
        logger.info("Run %s terminé : %s", run_id, summary["reason"])
        return {"run_id": run_id, "batch": name, "status": "SUCCESS", **summary}
    except DataValidationError:
        raise  # statut REJECTED déjà enregistré par l'étape de validation
    except Exception as exc:
        store.update_run(run_id, status="FAILED", message=str(exc)[:500])
        raise


@flow(name="pipeline-velo")
def pipeline_flow() -> list[dict]:
    """Traite tous les lots en attente, du plus ancien au plus récent."""
    logger = get_run_logger()
    batches = steps.list_incoming()
    if not batches:
        logger.info("Aucun nouveau lot.")
        return []
    results, failures = [], []
    for b in batches:
        try:
            results.append(process_batch(str(b)))
        except Exception as exc:  # un lot rejeté ne bloque pas les suivants
            failures.append(b.name)
            results.append({"batch": b.name, "status": "REJECTED"
                            if isinstance(exc, DataValidationError) else "FAILED",
                            "error": str(exc)})
    if failures:
        raise RuntimeError(f"Lots en échec : {failures}")
    return results


def watch(interval: int | None = None, max_loops: int | None = None) -> None:
    """Déclenchement automatique sans serveur : surveille le dossier incoming."""
    cfg = get_config()
    interval = interval or cfg.orchestration["watch_interval_seconds"]
    loops = 0
    log.info("Surveillance de %s toutes les %ss", rel(cfg.path("incoming_dir"), cfg), interval)
    while max_loops is None or loops < max_loops:
        if steps.list_incoming(cfg):
            try:
                pipeline_flow()
            except Exception as exc:
                log.error("Pipeline en échec : %s", exc)
        loops += 1
        if max_loops is None or loops < max_loops:  # pas d'attente inutile au dernier tour
            time.sleep(interval)


def serve(interval: int | None = None) -> None:
    """Déploiement Prefect planifié (nécessite `prefect server start`)."""
    cfg = get_config()
    pipeline_flow.serve(name="pipeline-velo-planifie",
                        interval=interval or cfg.orchestration["watch_interval_seconds"])
