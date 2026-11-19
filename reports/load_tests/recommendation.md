# Recommandation de dimensionnement

SLO retenu : p95 <= 200 ms et taux d'erreur <= 1%.

| Charge (utilisateurs) | Config. recommandée | Débit (req/s) | p95 (ms) | CPU moyen | RAM max |
|---|---|---|---|---|---|
| 10 | 1 worker(s), 1 vCPU | 53.1 | 160 | 77.4% | 384 Mo |
| 25 | aucune config. testée ne tient le SLO (meilleure : 2 worker(s), p95 360 ms) | 93.6 | 360 | 137.9% | 810 Mo |
| 50 | aucune config. testée ne tient le SLO (meilleure : 2 worker(s), p95 860 ms) | 91.9 | 860 | 141.1% | 823 Mo |

CPU en % d'un cœur (200 % = 2 cœurs pleinement utilisés). Le générateur de charge tourne sur la même machine que l'API : les chiffres sont donc pessimistes.
