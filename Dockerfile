# BehaviorDNA — one image, two entrypoints (API + dashboard) via docker-compose.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1: LightGBM runtime. git: DVC needs it for `dvc pull`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first for layer caching.
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code (models/data come via volume mount or `dvc pull` at startup — see
# docker/entrypoint.sh and .dockerignore).
COPY . .
RUN chmod +x docker/entrypoint.sh

EXPOSE 8000 8501

# entrypoint optionally `dvc pull`s artifacts when a DagsHub token is present,
# then execs the service command (overridden per service in docker-compose.yml).
ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
