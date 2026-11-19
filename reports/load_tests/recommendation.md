# Recommandation de dimensionnement

SLO retenu : p95 <= 200 ms et taux d'erreur <= 1%.

| Charge (utilisateurs) | Config. recommandée | Débit (req/s) | p95 (ms) | CPU moyen | RAM max |
|---|---|---|---|---|---|
| 10 | 1 worker(s), 1 vCPU | 53.2 | 150 | 78.4% | 327 Mo |
| 25 | aucune config. testée ne tient le SLO (meilleure : 2 worker(s), p95 280 ms) | 101.6 | 280 | 147.5% | 697 Mo |
| 50 | aucune config. testée ne tient le SLO (meilleure : 2 worker(s), p95 740 ms) | 91.7 | 740 | 140.1% | 709 Mo |

CPU en % d'un cœur (200 % = 2 cœurs pleinement utilisés). Le générateur de charge tourne sur la même machine que l'API : les chiffres sont donc pessimistes.
