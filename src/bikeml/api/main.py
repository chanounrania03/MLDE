"""API de prédiction (FastAPI).

Endpoints :
  GET  /health          état du service et version servie
  GET  /model           détails de la version servie
  POST /predict         prédiction unitaire
  POST /predict/batch   prédiction par lot (taille max configurable)
  POST /feedback        valeur réelle observée pour une prédiction passée
  POST /admin/reload    recharge le champion depuis le registre (protégé par jeton
                        si BIKEML_ADMIN_TOKEN est défini)
  GET  /metrics         métriques Prometheus

Chaque appel est tracé dans data/ops.db (table requests) avec sa latence,
son code retour, la version du modèle et l'erreur éventuelle. Chaque
prédiction est conservée avec ses entrées (table predictions), ce qui permet
de rapprocher les feedbacks et de calculer l'erreur réelle par version.
L'identifiant de requête est généré par le serveur (jamais repris du client, pour
qu'il reste unique : c'est la clé qui relie une prédiction à son feedback). Il est
renvoyé dans le corps et dans l'en-tête X-Request-ID.
"""
from __future__ import annotations

import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from bikeml.api.model_service import ModelService, ModelUnavailable
from bikeml.api.schemas import (BatchRequest, BatchResponse, BikeFeatures, FeedbackRequest,
                                PredictionResponse)
from bikeml.config import get_config
from bikeml.monitoring.store import OpsStore

log = logging.getLogger("bikeml.api")

REQUESTS = Counter("bikeml_requests_total", "Requêtes HTTP", ["endpoint", "status"])
LATENCY = Histogram("bikeml_request_latency_seconds", "Latence", ["endpoint"],
                    buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5))
PREDICTIONS = Counter("bikeml_predictions_total", "Prédictions servies", ["model_version"])

UNTRACKED = {"/metrics", "/docs", "/openapi.json", "/favicon.ico"}


def create_app(service: ModelService | None = None, store: OpsStore | None = None,
               load_on_startup: bool = True) -> FastAPI:
    cfg = get_config()
    service = service or ModelService(cfg)
    store = store or OpsStore(cfg.path("ops_db"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_on_startup and not service.ready:
            if not service.load_champion():
                store.alert("ERROR", "api", f"Démarrage dégradé : {service.last_error}")
        yield

    app = FastAPI(title="API demande de vélos", version="1.0.0", lifespan=lifespan)
    app.state.service, app.state.store = service, store

    # ---------------------------------------------------------------- traces
    @app.middleware("http")
    async def trace_requests(request: Request, call_next):
        # toujours généré ici : un identifiant fourni par le client pourrait être
        # réutilisé et rattacher un feedback à la mauvaise prédiction
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.error = None
        request.state.n_items = None
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # erreur non prévue : 500 propre + trace
            log.exception("Erreur interne")
            request.state.error = f"{type(exc).__name__}: {exc}"
            response = JSONResponse({"detail": "erreur interne", "request_id": request_id},
                                    status_code=500)
        latency = (time.perf_counter() - start) * 1000
        path = request.url.path
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Model-Version"] = str(service.version)
        if path not in UNTRACKED:
            REQUESTS.labels(path, response.status_code).inc()
            LATENCY.labels(path).observe(latency / 1000)
            try:
                store.log_request(request_id, path, response.status_code, latency,
                                  service.version, request.state.n_items, request.state.error)
            except Exception as exc:  # la traçabilité ne doit pas casser le service
                log.error("Trace non enregistrée : %s", exc)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        errors = [{"champ": ".".join(str(p) for p in e["loc"][1:]), "erreur": e["msg"]}
                  for e in exc.errors()]
        request.state.error = f"validation: {errors[:3]}"[:300]
        return JSONResponse({"detail": "entrée invalide", "errors": errors,
                             "request_id": request.state.request_id}, status_code=422)

    @app.exception_handler(HTTPException)
    async def http_handler(request: Request, exc: HTTPException):
        request.state.error = str(exc.detail)[:300]
        return JSONResponse({"detail": exc.detail, "request_id": request.state.request_id},
                            status_code=exc.status_code)

    def _predict(request: Request, rows: list[dict]) -> tuple[list[float], str]:
        request.state.n_items = len(rows)
        try:
            preds, version = service.predict(rows)
        except ModelUnavailable as exc:
            raise HTTPException(503, f"modèle indisponible : {exc}") from exc
        store.log_predictions(request.state.request_id, rows, preds, version)
        PREDICTIONS.labels(version).inc(len(preds))
        return preds, version

    # ------------------------------------------------------------- endpoints
    @app.get("/health")
    def health():
        return {"status": "ok" if service.ready else "degraded",
                "model_version": service.version, "loaded_at": service.loaded_at,
                "last_error": service.last_error}

    @app.get("/model")
    def model_info():
        if not service.ready:
            raise HTTPException(503, f"modèle indisponible : {service.last_error}")
        return {"name": cfg.model["name"], "version": service.version,
                "alias": cfg.promotion["champion_alias"], "loaded_at": service.loaded_at,
                "features": cfg.model["features"]}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(item: BikeFeatures, request: Request):
        t0 = time.perf_counter()
        preds, version = _predict(request, [item.model_dump()])
        return PredictionResponse(request_id=request.state.request_id, prediction=preds[0],
                                  model_version=version,
                                  latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, request: Request):
        if len(batch.items) > cfg.api["max_batch_size"]:
            raise HTTPException(413, f"lot trop grand (max {cfg.api['max_batch_size']})")
        t0 = time.perf_counter()
        preds, version = _predict(request, [i.model_dump() for i in batch.items])
        return BatchResponse(request_id=request.state.request_id, predictions=preds,
                             model_version=version,
                             latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    @app.post("/feedback")
    def feedback(fb: FeedbackRequest):
        found = store.add_feedback(fb.request_id, fb.item_index, fb.actual_cnt)
        if found is None:
            raise HTTPException(404, "prédiction inconnue pour ce request_id / item_index")
        if found.get("duplicate"):
            # un second retour pour la même prédiction fausserait l'erreur mesurée
            raise HTTPException(409, "feedback déjà enregistré pour cette prédiction")
        return {"status": "enregistré", "error": round(fb.actual_cnt - found["prediction"], 2),
                **found}

    @app.post("/admin/reload")
    def reload(x_admin_token: str | None = Header(default=None)):
        expected = os.environ.get("BIKEML_ADMIN_TOKEN")
        if expected and not hmac.compare_digest(x_admin_token or "", expected):
            raise HTTPException(401, "jeton d'administration absent ou invalide")
        before = service.version
        ok = service.load_champion()
        if not ok:
            store.alert("ERROR", "api", f"Rechargement impossible : {service.last_error}")
            raise HTTPException(503, f"rechargement impossible, version {before} conservée"
                                     f" ({service.last_error})")
        return {"previous_version": before, "model_version": service.version}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
