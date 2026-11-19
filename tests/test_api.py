"""Tests de l'API avec un modèle factice (aucune dépendance à MLflow)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from bikeml.api.main import create_app
from bikeml.api.model_service import ModelService
from bikeml.monitoring.store import OpsStore

SAMPLE = {"season": 3, "mnth": 7, "hr": 17, "holiday": 0, "weekday": 3, "workingday": 1,
          "weathersit": 1, "temp": 0.72, "atemp": 0.66, "hum": 0.45, "windspeed": 0.19}


class ConstantModel:
    def predict(self, X):
        return np.full(len(X), 123.0)


@pytest.fixture()
def api(cfg):
    service = ModelService(cfg)
    service.set_model(ConstantModel(), "7")
    store = OpsStore(cfg.path("ops_db"))
    with TestClient(create_app(service, store, load_on_startup=False)) as client:
        yield client, store, service


def test_health(api):
    client, _, _ = api
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["model_version"] == "7"


def test_prediction_unitaire_et_trace(api):
    client, store, _ = api
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 123.0 and body["model_version"] == "7"
    assert r.headers["X-Request-ID"] == body["request_id"]
    req = store.query("SELECT * FROM requests WHERE request_id=?", (body["request_id"],))
    assert req[0]["status_code"] == 200 and req[0]["model_version"] == "7"
    assert len(store.query("SELECT * FROM predictions")) == 1


def test_prediction_par_lot(api):
    client, _, _ = api
    r = client.post("/predict/batch", json={"items": [SAMPLE] * 3})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 3


def test_lot_trop_grand(api, cfg):
    client, _, _ = api
    n = cfg.api["max_batch_size"] + 1
    assert client.post("/predict/batch", json={"items": [SAMPLE] * n}).status_code == 413


@pytest.mark.parametrize("patch", [{"hr": 30}, {"temp": 3.0}, {"season": "été"},
                                   {"inconnu": 1}])
def test_entree_invalide(api, patch):
    client, store, _ = api
    r = client.post("/predict", json={**SAMPLE, **patch})
    assert r.status_code == 422 and r.json()["errors"]
    assert store.query("SELECT error FROM requests WHERE status_code=422")[0]["error"]


def test_champ_manquant(api):
    client, _, _ = api
    payload = dict(SAMPLE)
    payload.pop("hr")
    assert client.post("/predict", json=payload).status_code == 422


def test_feedback(api):
    client, store, _ = api
    rid = client.post("/predict", json=SAMPLE).json()["request_id"]
    r = client.post("/feedback", json={"request_id": rid, "actual_cnt": 130})
    assert r.status_code == 200 and r.json()["error"] == 7.0
    assert store.query("SELECT model_version FROM feedback")[0]["model_version"] == "7"
    assert client.post("/feedback", json={"request_id": "x", "actual_cnt": 1}).status_code == 404


def test_modele_indisponible(cfg):
    service = ModelService(cfg)  # rien de chargé : registre vide
    with TestClient(create_app(service, OpsStore(cfg.path("ops_db")))) as client:
        assert client.get("/health").json()["status"] == "degraded"
        r = client.post("/predict", json=SAMPLE)
        assert r.status_code == 503 and "indisponible" in r.json()["detail"]


def test_rechargement_echoue_conserve_la_version(api, monkeypatch):
    client, store, service = api
    from bikeml.model import registry

    def registre_ko(*args, **kwargs):
        raise ConnectionError("registre injoignable")

    monkeypatch.setattr(registry, "get_alias_version", registre_ko)
    r = client.post("/admin/reload")
    assert r.status_code == 503
    assert service.version == "7"                       # l'ancien modèle reste servi
    assert client.post("/predict", json=SAMPLE).status_code == 200
    assert client.get("/health").json()["last_error"]
    assert store.query("SELECT source FROM alerts")[0]["source"] == "api"


def test_request_id_genere_par_le_serveur(api):
    client, _, _ = api
    ids = {client.post("/predict", json=SAMPLE, headers={"X-Request-ID": "fixe"}).json()["request_id"]
           for _ in range(2)}
    assert len(ids) == 2 and "fixe" not in ids


def test_feedback_en_double_refuse(api):
    client, store, _ = api
    rid = client.post("/predict", json=SAMPLE).json()["request_id"]
    assert client.post("/feedback", json={"request_id": rid, "actual_cnt": 130}).status_code == 200
    assert client.post("/feedback", json={"request_id": rid, "actual_cnt": 99}).status_code == 409
    assert len(store.query("SELECT * FROM feedback")) == 1


def test_reload_protege_par_jeton(api, monkeypatch):
    client, _, service = api
    monkeypatch.setenv("BIKEML_ADMIN_TOKEN", "secret")
    assert client.post("/admin/reload").status_code == 401
    assert client.post("/admin/reload", headers={"X-Admin-Token": "faux"}).status_code == 401
    assert service.version == "7"
