"""Campagne de tests de charge et recommandation de dimensionnement.

Pour chaque configuration (nombre de workers uvicorn) et chaque niveau de
charge (utilisateurs simultanés), le script :
  1. démarre l'API sur un port dédié ;
  2. lance Locust en mode headless pendant la durée configurée ;
  3. mesure en parallèle le CPU et la mémoire de l'arbre de processus de l'API ;
  4. collecte débit, latences (p50/p95/p99) et taux d'erreur.

L'API testée écrit ses traces dans une base séparée (data/ops_loadtest.db) : le
trafic synthétique et ses feedbacks aléatoires ne polluent pas la supervision du
service réel (onglets Service et Qualité du dashboard).

Le résultat est écrit dans reports/load_tests/summary.csv, puis une
recommandation est déduite : la plus petite configuration qui respecte le SLO
(p95 et taux d'erreur, définis dans config.yaml) pour chaque niveau de charge.

Usage : python scripts/run_load_tests.py [--duration 30] [--workers 1 2] [--users 10 50]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bikeml.config import get_config  # noqa: E402

PORT = 8100


def wait_ready(url: str, timeout: float = 90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if b'"ok"' in r.read():
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("API non prête : le champion est-il bien enregistré ?")


class ResourceSampler(threading.Thread):
    """Échantillonne CPU (%) et RSS (Mo) du processus API et de ses enfants."""

    def __init__(self, pid: int):
        super().__init__(daemon=True)
        self.root = psutil.Process(pid)
        self.cpu, self.mem = [], []
        self._stop = threading.Event()
        self._known: dict[int, psutil.Process] = {}

    def procs(self) -> list[psutil.Process]:
        # On garde les mêmes objets Process d'un échantillon à l'autre : psutil
        # calcule le %CPU par différence avec l'appel précédent sur le même objet.
        try:
            for child in self.root.children(recursive=True):
                self._known.setdefault(child.pid, child)
        except psutil.NoSuchProcess:
            return []
        return [self.root, *self._known.values()]

    def run(self):
        for p in self.procs():
            p.cpu_percent(None)
        while not self._stop.wait(1.0):
            cpu = mem = 0.0
            for p in self.procs():
                try:
                    cpu += p.cpu_percent(None)
                    mem += p.memory_info().rss / 2**20
                except psutil.NoSuchProcess:
                    pass
            self.cpu.append(cpu)
            self.mem.append(mem)

    def stop(self):
        self._stop.set()


def stop_tree(pid: int) -> None:
    """Arrête l'API et ses workers (fonctionne sous Linux, macOS et Windows)."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = [*parent.children(recursive=True), parent]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=15)
    for p in alive:
        p.kill()


def read_aggregated(prefix: Path) -> dict:
    with open(f"{prefix}_stats.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["Name"] == "Aggregated":
                n = int(row["Request Count"])
                return {"requests": n,
                        "failures": int(row["Failure Count"]),
                        "error_rate": round(int(row["Failure Count"]) / max(n, 1), 4),
                        "rps": round(float(row["Requests/s"]), 1),
                        "p50_ms": float(row["50%"]), "p95_ms": float(row["95%"]),
                        "p99_ms": float(row["99%"])}
    raise RuntimeError(f"Statistiques introuvables pour {prefix}")


def run_campaign(workers_list, users_list, duration, spawn_rate, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["BIKEML_OPS_DB"] = str(get_config().root / "data" / "ops_loadtest.db")
    results = []
    for workers in workers_list:
        api = subprocess.Popen(
            [sys.executable, "-m", "bikeml", "api", "--workers", str(workers),
             "--port", str(PORT)], cwd=ROOT, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_ready(f"http://127.0.0.1:{PORT}/health")
            time.sleep(2)
            for users in users_list:
                prefix = out_dir / f"w{workers}_u{users}"
                print(f"-> {workers} worker(s), {users} utilisateurs, {duration}s", flush=True)
                sampler = ResourceSampler(api.pid)
                sampler.start()
                subprocess.run(
                    [sys.executable, "-m", "locust", "-f", str(ROOT / "load_tests" / "locustfile.py"), "--headless",
                     "-u", str(users), "-r", str(spawn_rate), "-t", f"{duration}s",
                     "--host", f"http://127.0.0.1:{PORT}", "--csv", str(prefix),
                     "--only-summary", "--loglevel", "ERROR"],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                sampler.stop()
                stats = read_aggregated(prefix)
                row = {"workers": workers, "users": users, **stats,
                       "cpu_mean_pct": round(statistics.mean(sampler.cpu or [0]), 1),
                       "cpu_max_pct": round(max(sampler.cpu or [0]), 1),
                       "ram_max_mb": round(max(sampler.mem or [0]), 1)}
                print("   ", row, flush=True)
                results.append(row)
        finally:
            stop_tree(api.pid)
            time.sleep(2)
    return results


def recommend(results: list[dict], slo_p95: float, slo_err: float) -> str:
    lines = ["# Recommandation de dimensionnement", "",
             f"SLO retenu : p95 <= {slo_p95:.0f} ms et taux d'erreur <= {slo_err:.0%}.", "",
             "| Charge (utilisateurs) | Config. recommandée | Débit (req/s) | p95 (ms) |"
             " CPU moyen | RAM max |",
             "|---|---|---|---|---|---|"]
    for users in sorted({r["users"] for r in results}):
        ok = [r for r in results if r["users"] == users
              and r["p95_ms"] <= slo_p95 and r["error_rate"] <= slo_err]
        if ok:
            best = min(ok, key=lambda r: (r["workers"], r["p95_ms"]))
            vcpu = max(1, math.ceil(best["cpu_max_pct"] / 100))
            lines.append(f"| {users} | {best['workers']} worker(s), {vcpu} vCPU |"
                         f" {best['rps']} | {best['p95_ms']:.0f} | {best['cpu_mean_pct']}% |"
                         f" {best['ram_max_mb']:.0f} Mo |")
        else:
            best = min((r for r in results if r["users"] == users), key=lambda r: r["p95_ms"])
            lines.append(f"| {users} | aucune config. testée ne tient le SLO (meilleure : "
                         f"{best['workers']} worker(s), p95 {best['p95_ms']:.0f} ms) | "
                         f"{best['rps']} | {best['p95_ms']:.0f} | {best['cpu_mean_pct']}% |"
                         f" {best['ram_max_mb']:.0f} Mo |")
    lines += ["", "CPU en % d'un cœur (200 % = 2 cœurs pleinement utilisés). "
              "Le générateur de charge tourne sur la même machine que l'API : "
              "les chiffres sont donc pessimistes."]
    return "\n".join(lines) + "\n"


def main() -> None:
    cfg = get_config()
    lt = cfg.load_test
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=lt["duration_seconds"])
    ap.add_argument("--workers", type=int, nargs="+", default=lt["workers"])
    ap.add_argument("--users", type=int, nargs="+", default=lt["users"])
    args = ap.parse_args()

    out_dir = cfg.path("reports_dir") / "load_tests"
    results = run_campaign(args.workers, args.users, args.duration, lt["spawn_rate"], out_dir)
    if not results:
        raise SystemExit("Aucune mesure collectée.")
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    text = recommend(results, lt["slo_p95_ms"], lt["slo_error_rate"])
    (out_dir / "recommendation.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
