"""Dashboard de supervision (Streamlit).

Lancement : streamlit run dashboard/app.py

Cinq onglets :
  - Traitements   : exécutions du pipeline, statut, lots rejetés, alertes ;
  - Modèles       : versions du registre, alias, décisions de mise en service ;
  - Service       : trafic, latences, codes retour, erreurs de l'API ;
  - Qualité       : erreur réelle mesurée grâce aux feedbacks, par version ;
  - Charge        : résultats de la dernière campagne de tests de charge.
Toutes les données viennent de data/ops.db, du registre MLflow et de reports/.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bikeml.config import get_config  # noqa: E402

cfg = get_config()
st.set_page_config(page_title="Supervision vélos", layout="wide")
st.title("Supervision - prévision de la demande de vélos")


@st.cache_data(ttl=10)
def q(sql: str) -> pd.DataFrame:
    db = cfg.path("ops_db")
    if not db.exists():
        return pd.DataFrame()
    with sqlite3.connect(db) as con:
        return pd.read_sql_query(sql, con)


@st.cache_data(ttl=30)
def versions() -> pd.DataFrame:
    try:
        from bikeml.model.registry import list_versions
        return pd.DataFrame(list_versions(cfg))
    except Exception as exc:
        st.warning(f"Registre MLflow injoignable : {exc}")
        return pd.DataFrame()


window = st.sidebar.selectbox("Fenêtre d'analyse du trafic",
                              ["1 heure", "24 heures", "7 jours", "tout"], index=3)
since = {"1 heure": "-1 hour", "24 heures": "-1 day", "7 jours": "-7 days"}.get(window)
where = f"WHERE ts >= strftime('%Y-%m-%dT%H:%M:%f', 'now', '{since}')" if since else ""
if st.sidebar.button("Rafraîchir"):
    st.cache_data.clear()

tabs = st.tabs(["Traitements", "Modèles", "Service", "Qualité", "Charge"])

# ------------------------------------------------------------------ traitements
with tabs[0]:
    runs = q("SELECT * FROM pipeline_runs ORDER BY started_at DESC")
    if runs.empty:
        st.info("Aucune exécution de pipeline enregistrée.")
    else:
        c = st.columns(4)
        c[0].metric("Exécutions", len(runs))
        c[1].metric("Succès", int((runs.status == "SUCCESS").sum()))
        c[2].metric("Lots rejetés", int((runs.status == "REJECTED").sum()))
        c[3].metric("Échecs techniques", int((runs.status == "FAILED").sum()))
        st.dataframe(runs, width="stretch", hide_index=True)
    st.subheader("Alertes")
    alerts = q("SELECT ts, level, source, message FROM alerts ORDER BY id DESC")
    st.dataframe(alerts, width="stretch", hide_index=True)

# ---------------------------------------------------------------------- modèles
with tabs[1]:
    v = versions()
    if not v.empty:
        champ = v[v.aliases.str.contains("champion")]
        st.metric("Version en service (champion)",
                  f"v{champ.version.iloc[0]}" if len(champ) else "aucune")
        st.line_chart(v.set_index("version")[["rmse", "baseline_rmse"]])
        st.caption("RMSE de chaque version sur ses 28 derniers jours de données, "
                   "comparée à la référence naïve.")
        st.dataframe(v, width="stretch", hide_index=True)

# ---------------------------------------------------------------------- service
with tabs[2]:
    req = q(f"SELECT * FROM requests {where}")
    if req.empty:
        st.info("Aucun appel à l'API sur la période.")
    else:
        req["ts"] = pd.to_datetime(req["ts"], format="ISO8601")
        ok = req[req.status_code < 400]
        c = st.columns(5)
        c[0].metric("Requêtes", len(req))
        c[1].metric("Taux d'erreur", f"{(req.status_code >= 400).mean():.1%}")
        c[2].metric("Latence p50", f"{ok.latency_ms.quantile(.5):.0f} ms" if len(ok) else "-")
        c[3].metric("Latence p95", f"{ok.latency_ms.quantile(.95):.0f} ms" if len(ok) else "-")
        c[4].metric("Prédictions servies",
                    int(q(f"SELECT COUNT(*) n FROM predictions {where}").n.iloc[0]))
        per_min = req.set_index("ts").resample("1min").size().rename("requêtes/min")
        st.subheader("Trafic")
        st.line_chart(per_min)
        st.subheader("Latence p95 par minute (ms)")
        st.line_chart(ok.set_index("ts").latency_ms.resample("1min").quantile(.95))
        left, right = st.columns(2)
        left.subheader("Codes retour")
        left.bar_chart(req.groupby(["endpoint", "status_code"]).size().unstack(fill_value=0))
        right.subheader("Dernières erreurs")
        right.dataframe(req[req.status_code >= 400].sort_values("ts", ascending=False)
                        [["ts", "endpoint", "status_code", "error"]].head(50),
                        hide_index=True, width="stretch")

# ---------------------------------------------------------------------- qualité
with tabs[3]:
    fb = q(f"SELECT * FROM feedback {where}")
    if fb.empty:
        st.info("Aucun feedback reçu sur la période.")
    else:
        fb["abs_err"] = (fb.actual - fb.prediction).abs()
        fb["sq_err"] = (fb.actual - fb.prediction) ** 2
        g = fb.groupby("model_version").agg(feedbacks=("id", "count"),
                                            mae=("abs_err", "mean"), mse=("sq_err", "mean"))
        g["rmse"] = g.mse ** 0.5
        st.dataframe(g.drop(columns="mse").round(1), width="stretch")
        st.caption("Erreur mesurée en production à partir des valeurs réelles renvoyées "
                   "par les utilisateurs.")

# ------------------------------------------------------------------------ charge
with tabs[4]:
    summary = cfg.path("reports_dir") / "load_tests" / "summary.csv"
    reco = cfg.path("reports_dir") / "load_tests" / "recommendation.md"
    if summary.exists():
        df = pd.read_csv(summary)
        st.dataframe(df, hide_index=True, width="stretch")
        st.bar_chart(df.assign(config=df.workers.astype(str) + "w/" + df.users.astype(str) + "u")
                     .set_index("config")[["p95_ms"]])
    if reco.exists():
        st.markdown(reco.read_text(encoding="utf-8"))
    if not summary.exists():
        st.info("Lancez `python scripts/run_load_tests.py` pour remplir cet onglet.")
