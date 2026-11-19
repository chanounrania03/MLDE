import numpy as np
import pandas as pd

from bikeml.model.registry import decide_promotion
from bikeml.model.train import NaiveBaseline, build_model, evaluate, temporal_split
from tests.conftest import make_hourly_data


def test_pas_de_fuite_de_cible(cfg):
    feats = cfg.model["features"]
    for leaky in ("cnt", "casual", "registered", "yr", "instant", "dteday", "mnth", "season"):
        assert leaky not in feats


def test_decoupage_temporel(cfg):
    df = make_hourly_data(days=30)
    df["dteday"] = pd.to_datetime(df["dteday"])
    train, test = temporal_split(df, 7)
    assert train["dteday"].max() < test["dteday"].min()
    assert test["dteday"].nunique() == 7


def test_modele_bat_la_reference(cfg):
    df = make_hourly_data(days=90)
    df["dteday"] = pd.to_datetime(df["dteday"])
    train, test = temporal_split(df, 14)
    feats, target = cfg.model["features"], cfg.model["target"]
    model = build_model(cfg).fit(train[feats], train[target])
    m = evaluate(model, test, cfg)
    b = evaluate(NaiveBaseline().fit(train[feats], train[target]), test, cfg)
    assert m["rmse"] < b["rmse"]
    assert np.all(model.predict(test[feats]) > -1)


def test_regles_de_promotion():
    assert decide_promotion(50, None, 200, 0.02).promote
    assert not decide_promotion(250, None, 200, 0.02).promote          # garde-fou
    assert not decide_promotion(80, None, 200, 0.02, baseline_rmse=70).promote
    assert decide_promotion(101, 100, 200, 0.02).promote                # dans la tolérance
    assert not decide_promotion(103, 100, 200, 0.02).promote            # dégradation


def test_poids_recents(cfg):
    from bikeml.model.train import recency_weights
    df = make_hourly_data(days=61)
    df["dteday"] = pd.to_datetime(df["dteday"])
    w = recency_weights(df, 30)
    assert w.max() == 1.0 and abs(w.min() - 0.25) < 1e-9
    assert recency_weights(df, None) is None
