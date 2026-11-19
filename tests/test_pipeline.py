"""Test d'intégration : étapes du pipeline + registre MLflow (temporaire)."""
import pytest

from bikeml.data.validate import DataValidationError
from bikeml.model import registry
from bikeml.monitoring.store import OpsStore
from bikeml.orchestration import steps
from tests.conftest import make_hourly_data


def run_one(cfg, df, name):
    path = cfg.path("incoming_dir") / name
    df.to_csv(path, index=False)
    store = OpsStore(cfg.path("ops_db"))
    store.start_run(name, name)
    steps.validate_step(path, name, cfg)
    steps.accept_step(path, cfg)
    result = steps.train_step(name, name, cfg)
    return steps.promotion_step(result, name, cfg)


def test_chaine_complete_promotion_et_rollback(cfg):
    first = run_one(cfg, make_hourly_data("2011-01-01", 60, seed=1), "a.csv")
    assert first["promoted"] and registry.get_alias_version("champion", cfg) == "1"

    second = run_one(cfg, make_hourly_data("2011-03-02", 30, seed=2), "b.csv")
    assert second["version"] == "2"
    if second["promoted"]:
        assert registry.get_alias_version("previous", cfg) == "1"
        assert registry.rollback(cfg=cfg) == "1"
        assert registry.get_alias_version("champion", cfg) == "1"

    versions = registry.list_versions(cfg)
    assert [v["version"] for v in versions] == [1, 2]


def test_lot_invalide_bloque_le_pipeline(cfg):
    bad = make_hourly_data(days=10)
    bad.loc[0:5, "hr"] = 40
    path = cfg.path("incoming_dir") / "bad.csv"
    bad.to_csv(path, index=False)
    store = OpsStore(cfg.path("ops_db"))
    store.start_run("r1", "bad.csv")
    with pytest.raises(DataValidationError):
        steps.validate_step(path, "r1", cfg)
    assert (cfg.path("quarantine_dir") / "bad.csv").exists()
    assert not list(cfg.path("accepted_dir").glob("*.csv"))
    assert store.query("SELECT status FROM pipeline_runs")[0]["status"] == "REJECTED"
    assert store.query("SELECT source FROM alerts")[0]["source"] == "validation"


def test_double_rollback_ne_remet_pas_la_version_retiree(cfg):
    run_one(cfg, make_hourly_data("2011-01-01", 60, seed=1), "a.csv")
    run_one(cfg, make_hourly_data("2011-03-02", 30, seed=2), "b.csv")
    c = registry.client(cfg)
    # on force un historique connu : v1 puis v2 champions successifs
    registry.promote("1", "test", cfg)
    registry.promote("2", "test", cfg)
    assert registry.rollback(cfg=cfg) == "1"
    assert c.get_model_version("bike-demand", "2").tags["status"] == "rolled_back"
    assert "rollback depuis v2" in c.get_model_version("bike-demand", "1").tags["decision"]
    # plus aucune version précédente valide : un second rollback est refusé
    # au lieu de remettre en service la v2 retirée
    with pytest.raises(RuntimeError):
        registry.rollback(cfg=cfg)
    assert registry.get_alias_version("champion", cfg) == "1"


def test_rollback_sans_version_precedente(cfg):
    run_one(cfg, make_hourly_data("2011-01-01", 40, seed=3), "a.csv")
    with pytest.raises(RuntimeError):
        registry.rollback(cfg=cfg)
