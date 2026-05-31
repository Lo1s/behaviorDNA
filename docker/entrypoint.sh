#!/usr/bin/env bash
# Optionally fetch DVC-tracked models/data when a DagsHub token is provided
# (e.g. on a fresh cloud build), then run the service command. Local runs that
# mount ./models + ./data as volumes skip the pull entirely.
set -e

if [ -n "$DAGSHUB_TOKEN" ] && [ ! -f models/model.pkl ]; then
  echo "[entrypoint] DAGSHUB_TOKEN set and models/model.pkl missing → dvc pull"
  dvc remote modify origin --local auth basic || true
  dvc remote modify origin --local user "${DAGSHUB_USER:-token}" || true
  dvc remote modify origin --local password "$DAGSHUB_TOKEN" || true
  dvc pull -q || echo "[entrypoint] dvc pull failed — endpoints will degrade gracefully"
fi

exec "$@"
