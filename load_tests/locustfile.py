"""Scénario de charge Locust.

Profil d'usage simulé (pondérations) :
  - 10 : prédiction unitaire (cas majoritaire : une application mobile)
  -  2 : prédiction par lot de 24 éléments (planning d'une journée)
  -  1 : envoi d'un feedback sur la dernière prédiction (une seule fois : l'API
         refuse un second retour pour la même prédiction)
  -  1 : contrôle de santé

Les entrées sont tirées aléatoirement dans les domaines valides.
"""
import random

from locust import HttpUser, between, task


def random_features() -> dict:
    mnth = random.randint(1, 12)
    weekday = random.randint(0, 6)
    temp = round(random.uniform(0.05, 0.95), 2)
    return {"season": (mnth % 12) // 3 + 1, "mnth": mnth, "hr": random.randint(0, 23),
            "holiday": 0, "weekday": weekday, "workingday": int(1 <= weekday <= 5),
            "weathersit": random.choice([1, 1, 1, 2, 2, 3]), "temp": temp,
            "atemp": round(temp * 0.95, 2), "hum": round(random.uniform(0.2, 0.95), 2),
            "windspeed": round(random.uniform(0, 0.5), 2)}


class BikeApiUser(HttpUser):
    wait_time = between(0.05, 0.2)

    def on_start(self):
        self.last_request_id = None

    @task(10)
    def predict(self):
        with self.client.post("/predict", json=random_features(), name="/predict",
                              catch_response=True) as r:
            if r.status_code == 200:
                self.last_request_id = r.json()["request_id"]
            else:
                r.failure(f"HTTP {r.status_code}")

    @task(2)
    def predict_batch(self):
        items = [random_features() for _ in range(24)]
        self.client.post("/predict/batch", json={"items": items}, name="/predict/batch")

    @task(1)
    def feedback(self):
        if self.last_request_id:
            self.client.post("/feedback", json={"request_id": self.last_request_id,
                                                "actual_cnt": random.randint(0, 600)},
                             name="/feedback")
            self.last_request_id = None

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
