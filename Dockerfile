# Image unique du projet : elle sert à l'API, au pipeline, au dashboard et au
# serveur MLflow (seule la commande change, voir docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # un worker = un cœur : évite que les threads OpenMP se disputent le CPU
    OMP_NUM_THREADS=1

WORKDIR /app

# Dépendances d'abord (couche mise en cache tant que requirements.txt ne change pas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code du projet, puis installation du paquet bikeml
COPY . .
RUN pip install --no-cache-dir -e . --no-deps

# API, dashboard, serveur MLflow
EXPOSE 8000 8501 5000

# Par défaut : l'API, à l'écoute de toutes les interfaces du conteneur
CMD ["python", "-m", "bikeml", "api", "--host", "0.0.0.0"]
