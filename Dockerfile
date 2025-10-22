# ===============================
# Dockerfile — Backend FastAPI (SQLite)
# Base: Debian (python:3.12-slim)
# ===============================

# ---------- Build stage ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (runtime libs only for build resolution):
# - libgl1        : OpenCV runtime (replaces old libgl1-mesa-glx)
# - libglib2.0-0  : OpenCV runtime dependency
# - libgomp1      : XGBoost runtime (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps into a relocatable prefix to copy later
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------- Final runtime stage ----------
FROM python:3.12-slim

WORKDIR /app

# Runtime libs (same as in builder)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Bring in Python site-packages from builder
COPY --from=builder /install /usr/local

# Copy the whole project (includes your dev.db at repo root)
COPY . /app

# Optional: verify that the SQLite DB is present at build-time
# (won't fail the build if it isn't)
RUN ls -lh dev.db || echo "⚠️  dev.db not found at build time (will still run if app creates it)."

# Expose API port
EXPOSE 8000

# Start FastAPI via Uvicorn
# Adjust module path if your entrypoint differs (e.g., app.api:app)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
