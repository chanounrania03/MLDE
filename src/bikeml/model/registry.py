"""Registre de modèles : règles de mise en service et retour arrière.

Conventions dans le registre MLflow (modèle `bike-demand`) :
  - alias `champion` : version servie par l'API ;
  - alias `previous` : champion précédent, cible naturelle d'un rollback ;
  - tag `status` sur chaque version : candidate / champion / rejected / archived ;
  - tag `decision` : justification lisible de la décision.

Règle de promotion (fonction pure `decide_promotion`, testée unitairement) :
  1. garde-fou absolu : RMSE sous un plafond (détecte un modèle aberrant) ;
  2. le challenger doit battre la référence naïve (moyenne par heure/jour ouvré) ;
  3. s'il existe un champion, le challenger doit faire au moins aussi bien que
     lui, à une tolérance près, SUR LE MÊME jeu d'évaluation (le plus récent).
Toutes les comparaisons se font sur la même période d'évaluation, sinon elles
n'ont pas de sens (la demande d'été est bien plus forte que celle d'hiver).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from bikeml.config import Config, get_config
from bikeml.model.train import evaluate, setup_mlflow

log = logging.getLogger("bikeml.registry")


@dataclass
class Decision:
    promote: bool
    reason: str


def decide_promotion(challenger_rmse: float, champion_rmse: float | None,
                     max_rmse: float, tolerance: float,
                     baseline_rmse: float | None = None) -> Decision:
    if challenger_rmse > max_rmse:
        return Decision(False, f"RMSE {challenger_rmse:.1f} > seuil absolu {max_rmse}")
    if baseline_rmse is not None and challenger_rmse >= baseline_rmse:
        return Decision(False, f"RMSE {challenger_rmse:.1f} ne bat pas la référence naïve"
                               f" ({baseline_rmse:.1f})")
    if champion_rmse is None:
        return Decision(True, f"premier modèle valide (RMSE {challenger_rmse:.1f})")
    limit = champion_rmse * (1 + tolerance)
    if challenger_rmse <= limit:
        return Decision(True, f"RMSE {challenger_rmse:.1f} <= champion {champion_rmse:.1f}"
                              f" x {1 + tolerance:.2f}")
    return Decision(False, f"RMSE {challenger_rmse:.1f} > champion {champion_rmse:.1f}"
                           f" x {1 + tolerance:.2f}")


def client(cfg: Config | None = None) -> MlflowClient:
    cfg = cfg or get_config()
    setup_mlflow(cfg)
    return MlflowClient()


def get_alias_version(alias: str, cfg: Config | None = None) -> str | None:
    cfg = cfg or get_config()
    try:
        return str(client(cfg).get_model_version_by_alias(cfg.model["name"], alias).version)
    except MlflowException as exc:
        # alias ou modèle absent -> None ; registre injoignable -> on propage
        if exc.error_code in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE",
                              "NOT_FOUND"):
            return None
        raise


def load_version(version: str, cfg: Config | None = None):
    cfg = cfg or get_config()
    setup_mlflow(cfg)
    return mlflow.sklearn.load_model(f"models:/{cfg.model['name']}/{version}")


def evaluate_and_decide(challenger_version: str, holdout, cfg: Config | None = None,
                        baseline_rmse: float | None = None
                        ) -> tuple[Decision, dict, dict | None]:
    """Compare challenger et champion sur le même jeu d'évaluation."""
    cfg = cfg or get_config()
    chall = evaluate(load_version(challenger_version, cfg), holdout, cfg)
    champ_version = get_alias_version(cfg.promotion["champion_alias"], cfg)
    champ = None
    if champ_version and champ_version != challenger_version:
        champ = evaluate(load_version(champ_version, cfg), holdout, cfg)
        champ["version"] = champ_version
    decision = decide_promotion(chall["rmse"], champ["rmse"] if champ else None,
                                cfg.promotion["max_rmse"], cfg.promotion["tolerance"],
                                baseline_rmse)
    return decision, chall, champ


def promote(version: str, reason: str, cfg: Config | None = None) -> None:
    cfg = cfg or get_config()
    c, name = client(cfg), cfg.model["name"]
    champ_alias, prev_alias = cfg.promotion["champion_alias"], cfg.promotion["previous_alias"]
    old = get_alias_version(champ_alias, cfg)
    if old and old != version:
        c.set_registered_model_alias(name, prev_alias, old)
        c.set_model_version_tag(name, old, "status", "archived")
    c.set_registered_model_alias(name, champ_alias, version)
    c.set_model_version_tag(name, version, "status", "champion")
    c.set_model_version_tag(name, version, "decision", reason)
    log.info("Version %s promue champion (%s)", version, reason)


def reject(version: str, reason: str, cfg: Config | None = None) -> None:
    cfg = cfg or get_config()
    c = client(cfg)
    c.set_model_version_tag(cfg.model["name"], version, "status", "rejected")
    c.set_model_version_tag(cfg.model["name"], version, "decision", reason)
    log.info("Version %s rejetée (%s)", version, reason)


def rollback(to_version: str | None = None, cfg: Config | None = None) -> str:
    """Remet en service `to_version`, ou par défaut le champion précédent."""
    cfg = cfg or get_config()
    target = to_version or get_alias_version(cfg.promotion["previous_alias"], cfg)
    if not target:
        raise RuntimeError("Aucune version précédente disponible pour un retour arrière.")
    current = get_alias_version(cfg.promotion["champion_alias"], cfg)
    c, name = client(cfg), cfg.model["name"]
    mv = c.get_model_version(name, target)  # lève une erreur si la version n'existe pas
    c.set_registered_model_alias(name, cfg.promotion["champion_alias"], target)
    c.set_model_version_tag(name, target, "status", "champion")
    # on garde la justification d'origine et on y ajoute la bascule (traçabilité)
    origin = mv.tags.get("decision", "")
    note = f"rollback depuis v{current}"
    c.set_model_version_tag(name, target, "decision", f"{origin} | {note}" if origin else note)
    if current and current != target:
        c.set_model_version_tag(name, current, "status", "rolled_back")
    # @previous ne doit jamais désigner la version qu'on vient de retirer : sinon un
    # second `rollback` la remettrait en service. Il pointe vers le dernier ancien
    # champion antérieur à la cible (statut archived), ou disparaît s'il n'y en a pas.
    fallback = _last_archived_before(target, cfg)
    if fallback:
        c.set_registered_model_alias(name, cfg.promotion["previous_alias"], fallback)
    elif get_alias_version(cfg.promotion["previous_alias"], cfg):
        c.delete_registered_model_alias(name, cfg.promotion["previous_alias"])
    log.warning("Rollback : champion v%s -> v%s (previous -> %s)", current, target,
                f"v{fallback}" if fallback else "aucune")
    return target


def _last_archived_before(version: str, cfg: Config) -> str | None:
    c, name = client(cfg), cfg.model["name"]
    older = [int(mv.version) for mv in c.search_model_versions(f"name='{name}'")
             if int(mv.version) < int(version) and mv.tags.get("status") == "archived"]
    return str(max(older)) if older else None


def list_versions(cfg: Config | None = None) -> list[dict]:
    cfg = cfg or get_config()
    c, name = client(cfg), cfg.model["name"]
    rows = []
    for mv in c.search_model_versions(f"name='{name}'"):
        run = c.get_run(mv.run_id)
        aliases = c.get_model_version(name, mv.version).aliases
        rows.append({
            "version": int(mv.version),
            "aliases": ",".join(aliases),
            "status": mv.tags.get("status", ""),
            "rmse": round(run.data.metrics.get("rmse", float("nan")), 2),
            "baseline_rmse": round(run.data.metrics.get("baseline_rmse", float("nan")), 2),
            "trigger_batch": run.data.params.get("trigger_batch"),
            "decision": mv.tags.get("decision", ""),
        })
    return sorted(rows, key=lambda r: r["version"])
