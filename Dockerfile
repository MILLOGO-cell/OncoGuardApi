# ===============================
# Dockerfile pour le backend FastAPI
# ===============================

# Étape 1 : build
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installer dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances Python
COPY requirements.txt .

# Installer les dépendances dans un dossier temporaire
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Étape 2 : runtime final
FROM python:3.12-slim

WORKDIR /app

# Copier les dépendances Python depuis l’étape précédente
COPY --from=builder /install /usr/local

# Copier tout le code source du projet
COPY . /app

# Copier la base SQLite (elle est déjà dans le dépôt à la racine)
# —> rien à changer, elle sera copiée automatiquement avec le COPY ci-dessus

# Vérifie que la base est bien accessible
RUN ls -lh dev.db || echo "⚠️ Base de données SQLite absente, vérifier le dépôt."

# Exposer le port du backend
EXPOSE 8000

# Commande de démarrage
# NB : ta structure de projet indique que ton app principale est dans /app/app/
# Ajuste le chemin du module selon ton fichier principal (main.py, api.py, etc.)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
