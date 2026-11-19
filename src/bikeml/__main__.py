"""Interface en ligne de commande : `python -m bikeml <commande>`.

    setup           télécharge les données et prépare les lots de rejeu
    release         dépose le prochain lot historique dans data/incoming
    release-bad     dépose un lot volontairement corrompu (incident contrôlé)
    run             exécute le pipeline sur tous les lots en attente
    retrain         réentraîne à la demande (--set max_iter=50 ...) avec les mêmes règles
    watch           surveille data/incoming et déclenche le pipeline automatiquement
    serve           déploiement Prefect planifié (serveur Prefect requis)
    models          liste les versions du registre et leur statut
    rollback        remet en service la version précédente (ou --version N)
    api             lance l'API de prédiction
"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s")
    for noisy in ("httpx", "alembic", "mlflow.store", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(prog="bikeml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    sub.add_parser("release")
    sub.add_parser("release-bad")
    sub.add_parser("run")
    w = sub.add_parser("watch")
    w.add_argument("--interval", type=int)
    w.add_argument("--max-loops", type=int)
    s = sub.add_parser("serve")
    s.add_argument("--interval", type=int)
    t = sub.add_parser("retrain")
    t.add_argument("--set", nargs="*", default=[], metavar="CLE=VALEUR",
                   help="surcharge d'hyperparamètres, ex. --set max_iter=10")
    sub.add_parser("models")
    r = sub.add_parser("rollback")
    r.add_argument("--version")
    a = sub.add_parser("api")
    a.add_argument("--workers", type=int, default=1)
    a.add_argument("--port", type=int)
    args = parser.parse_args(argv)

    if args.cmd == "setup":
        from bikeml.data.ingest import download_raw, prepare_batches
        download_raw()
        print(f"{len(prepare_batches())} lots prêts dans data/batches")
    elif args.cmd == "release":
        from bikeml.config import rel
        from bikeml.data.ingest import release_next_batch
        p = release_next_batch()
        print(rel(p) if p else "plus de lot à rejouer")
    elif args.cmd == "release-bad":
        from bikeml.config import rel
        from bikeml.data.ingest import make_invalid_batch
        print(rel(make_invalid_batch()))
    elif args.cmd == "run":
        from bikeml.orchestration.flows import pipeline_flow
        try:
            print(json.dumps(pipeline_flow(), indent=2, default=str, ensure_ascii=False))
        except Exception as exc:
            print(f"Pipeline en échec : {exc}", file=sys.stderr)
            return 1
    elif args.cmd == "watch":
        from bikeml.orchestration.flows import watch
        watch(args.interval, args.max_loops)
    elif args.cmd == "serve":
        from bikeml.orchestration.flows import serve
        serve(args.interval)
    elif args.cmd == "retrain":
        from bikeml.orchestration.steps import manual_retrain
        import yaml
        override = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
        summary = manual_retrain(override)
        print(json.dumps({k: summary[k] for k in ("version", "promoted", "reason")},
                         ensure_ascii=False))
    elif args.cmd == "models":
        from bikeml.model.registry import list_versions
        for row in list_versions():
            print(row)
    elif args.cmd == "rollback":
        from bikeml.model.registry import rollback
        from bikeml.orchestration.steps import notify_api
        from bikeml.config import get_config
        print(f"champion -> v{rollback(args.version)}")
        notify_api(get_config())
    elif args.cmd == "api":
        import os
        # Un worker = un cœur. Sans cette limite, chaque prédiction du gradient
        # boosting lance un pool OpenMP sur tous les cœurs : les requêtes
        # simultanées se disputent les CPU (mesuré : CPU x2 et p95 plus élevée).
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        import uvicorn
        from bikeml.config import get_config
        cfg = get_config()
        uvicorn.run("bikeml.api.main:app", host=cfg.api["host"],
                    port=args.port or cfg.api["port"], workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
