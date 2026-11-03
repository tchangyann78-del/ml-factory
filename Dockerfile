# syntax=docker/dockerfile:1

###############################################################################
# Stage 1 - builder : installe les dépendances et entraîne le modèle
###############################################################################
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Environnement virtuel isolé : il sera copié tel quel dans l'image finale
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 1) Les dépendances d'abord : cette couche reste en cache tant que
#    requirements.txt ne change pas (alors que le code change à chaque commit)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Le modèle est un artefact de build, produit avec les MÊMES versions que le runtime
COPY scripts/ scripts/
RUN python scripts/train.py

###############################################################################
# Stage 2 - runner : image finale minimale, sans outils de build
###############################################################################
FROM python:3.12-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:$PATH"

# Utilisateur système non-root : UID numérique fixe, sans shell ni répertoire personnel
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Les fichiers restent propriété de root : l'application peut les lire, pas les modifier
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/models ./models
COPY src/ ./src/

USER app
EXPOSE 8000

# python:slim n'a pas curl : on utilise l'interpréteur pour sonder /health
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

# Forme "exec" (JSON) : uvicorn est le PID 1 et reçoit directement SIGTERM -> arrêt propre
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]  