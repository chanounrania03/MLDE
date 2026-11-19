"""Persistance des traces opérationnelles dans SQLite.

Une seule base (data/ops.db) regroupe :
  - pipeline_runs : chaque exécution du pipeline (statut, lot, durée, erreur) ;
  - alerts        : alertes émises (lot rejeté, échec d'étape, dépendance KO) ;
  - requests      : chaque appel HTTP à l'API (endpoint, code, latence, version) ;
  - predictions   : chaque prédiction servie avec ses entrées ;
  - feedback      : valeurs réelles renvoyées après coup par les utilisateurs.

SQLite suffit pour un service local ; le mode WAL permet plusieurs lecteurs
(dashboard) pendant que l'API écrit.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,            -- RUNNING / SUCCESS / REJECTED / FAILED / SKIPPED
    batch TEXT,
    n_rows INTEGER,
    step TEXT,                       -- dernière étape atteinte
    model_version TEXT,
    promoted INTEGER,
    rmse REAL,
    message TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    source TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    model_version TEXT,
    n_items INTEGER,
    error TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    item_index INTEGER NOT NULL,
    ts TEXT NOT NULL,
    features TEXT NOT NULL,
    prediction REAL NOT NULL,
    model_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_req ON predictions(request_id);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    request_id TEXT NOT NULL,
    item_index INTEGER NOT NULL,
    actual REAL NOT NULL,
    prediction REAL,
    model_version TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class OpsStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ---- pipeline -------------------------------------------------------
    def start_run(self, run_id: str, batch: str | None) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO pipeline_runs(run_id, started_at, status, batch, step)"
                " VALUES (?,?,?,?,?)", (run_id, now(), "RUNNING", batch, "start"))

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        if fields.get("status") in {"SUCCESS", "REJECTED", "FAILED", "SKIPPED"}:
            fields.setdefault("finished_at", now())
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.connect() as con:
            con.execute(f"UPDATE pipeline_runs SET {cols} WHERE run_id=?",
                        (*fields.values(), run_id))

    def alert(self, level: str, source: str, message: str) -> None:
        with self.connect() as con:
            con.execute("INSERT INTO alerts(ts, level, source, message) VALUES (?,?,?,?)",
                        (now(), level, source, message))

    # ---- API ------------------------------------------------------------
    def log_request(self, request_id: str, endpoint: str, status_code: int,
                    latency_ms: float, model_version: str | None,
                    n_items: int | None = None, error: str | None = None) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO requests VALUES (?,?,?,?,?,?,?,?)",
                (request_id, now(), endpoint, status_code, round(latency_ms, 3),
                 model_version, n_items, error))

    def log_predictions(self, request_id: str, rows: list[dict], preds: list[float],
                        model_version: str) -> None:
        ts = now()
        with self.connect() as con:
            con.executemany(
                "INSERT INTO predictions(request_id, item_index, ts, features, prediction,"
                " model_version) VALUES (?,?,?,?,?,?)",
                [(request_id, i, ts, json.dumps(r), float(p), model_version)
                 for i, (r, p) in enumerate(zip(rows, preds))])

    def add_feedback(self, request_id: str, item_index: int, actual: float) -> dict | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT prediction, model_version FROM predictions"
                " WHERE request_id=? AND item_index=?", (request_id, item_index)).fetchone()
            if row is None:
                return None
            if con.execute("SELECT 1 FROM feedback WHERE request_id=? AND item_index=?",
                           (request_id, item_index)).fetchone():
                return {"duplicate": True}
            con.execute(
                "INSERT INTO feedback(ts, request_id, item_index, actual, prediction,"
                " model_version) VALUES (?,?,?,?,?,?)",
                (now(), request_id, item_index, actual, row["prediction"],
                 row["model_version"]))
            return {"prediction": row["prediction"], "model_version": row["model_version"]}

    # ---- lecture (dashboard / notebook) ---------------------------------
    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self.connect() as con:
            return [dict(r) for r in con.execute(sql, params).fetchall()]
