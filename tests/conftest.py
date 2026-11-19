"""Fixtures communes : un projet isolé dans un dossier temporaire.

Les tests n'utilisent ni le réseau ni les données réelles : un petit jeu
synthétique au format UCI est généré, et MLflow écrit dans une base SQLite
temporaire. Les tests sont donc rapides et reproductibles.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bikeml import config as config_module

ROOT = Path(__file__).resolve().parents[1]


def make_hourly_data(start: str = "2011-01-01", days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    df = pd.DataFrame({"dteday": idx.strftime("%Y-%m-%d"), "hr": idx.hour})
    d = pd.to_datetime(df["dteday"])
    df["mnth"] = d.dt.month
    df["season"] = ((df["mnth"] % 12) // 3) + 1
    df["yr"] = (d.dt.year - 2011).clip(0, 1)
    df["weekday"] = (d.dt.dayofweek + 1) % 7
    df["holiday"] = 0
    df["workingday"] = df["weekday"].between(1, 5).astype(int)
    df["weathersit"] = rng.integers(1, 4, len(df))
    df["temp"] = rng.uniform(0.1, 0.9, len(df)).round(2)
    df["atemp"] = (df["temp"] * 0.95).round(4)
    df["hum"] = rng.uniform(0.2, 0.9, len(df)).round(2)
    df["windspeed"] = rng.uniform(0, 0.5, len(df)).round(4)
    peak = np.where(df["workingday"] == 1,
                    np.exp(-((df["hr"] - 8) ** 2) / 4) + np.exp(-((df["hr"] - 17) ** 2) / 4),
                    np.exp(-((df["hr"] - 14) ** 2) / 10))
    base = 20 + 300 * peak * (0.5 + df["temp"]) / df["weathersit"]
    df["cnt"] = rng.poisson(base).astype(int)
    df["casual"] = (df["cnt"] * 0.2).astype(int)
    df["registered"] = df["cnt"] - df["casual"]
    cols = ["dteday", "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
            "weathersit", "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
    return df[cols]


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    raw = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    raw["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
    raw["model"]["params"]["max_iter"] = 50
    raw["model"]["holdout_days"] = 7
    (tmp_path / "configs").mkdir()
    cfg_file = tmp_path / "configs" / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("BIKEML_CONFIG", str(cfg_file))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("BIKEML_API_URL", "http://127.0.0.1:9")  # API volontairement absente
    monkeypatch.chdir(tmp_path)
    config_module._load.cache_clear()
    c = config_module.get_config()
    config_module.ensure_dirs(c)
    yield c
    config_module._load.cache_clear()
    shutil.rmtree(tmp_path / "mlruns", ignore_errors=True)


@pytest.fixture()
def good_batch() -> pd.DataFrame:
    return make_hourly_data()
