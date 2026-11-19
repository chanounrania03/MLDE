"""Émission d'alertes.

Une alerte est toujours :
  1. écrite dans les logs ;
  2. enregistrée dans la table `alerts` (visible dans le dashboard) ;
  3. envoyée à un webhook si ALERT_WEBHOOK_URL est défini (Slack, Discord,
     Teams... acceptent tous un POST JSON). L'URL reste hors du dépôt.

L'envoi webhook ne doit jamais faire échouer le pipeline : une erreur réseau
est seulement journalisée.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

from bikeml.config import get_config
from bikeml.monitoring.store import OpsStore

log = logging.getLogger("bikeml.alerts")


def send_alert(message: str, level: str = "ERROR", source: str = "pipeline") -> None:
    message = message if len(message) <= 500 else message[:497] + "..."
    log.log(logging.ERROR if level == "ERROR" else logging.WARNING, "[ALERTE] %s", message)
    try:
        OpsStore(get_config().path("ops_db")).alert(level, source, message)
    except Exception as exc:  # la base elle-même peut être la cause du problème
        log.error("Impossible d'enregistrer l'alerte en base : %s", exc)

    url = os.environ.get("ALERT_WEBHOOK_URL")
    if not url:
        return
    payload = json.dumps({"text": f"[{level}] {source} : {message}",
                          "content": f"[{level}] {source} : {message}"}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:
        log.warning("Webhook d'alerte injoignable : %s", exc)
