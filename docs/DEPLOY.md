# Deployment

Two surfaces ship: the **inference API** (FastAPI) and the **dashboard**
(Streamlit). Both run from one Docker image; the dashboard can also be hosted
free on Streamlit Community Cloud.

> Artifacts (`models/`, `data/`) are **DVC-tracked**, not in git. Every path
> below either mounts what `dvc repro` produced locally or `dvc pull`s with a
> DagsHub token. Both the API and dashboard **degrade gracefully** when the model
> is absent (the API serves `/health`; the dashboard shows a provisioning note).

---

## Local — Docker Compose (API + dashboard)

```bash
dvc repro                       # produce models/ + data/splits/ locally (once)
docker compose up --build
#  API        → http://localhost:8000   (interactive docs at /docs)
#  Dashboard  → http://localhost:8501
```

`docker-compose.yml` mounts `./models`, `./data`, `./reports` into both
services, so the containers use exactly what your pipeline produced — no network
needed. On a host with **no** local artifacts, set `DAGSHUB_USER` /
`DAGSHUB_TOKEN` first and the entrypoint (`docker/entrypoint.sh`) will `dvc pull`
them at startup instead.

> The image installs the full `requirements.txt` (incl. PyTorch), so it's large.
> The API `/predict` endpoints and dashboard tabs 1–4 don't need torch — a slim
> API-only image (drop torch + the streaming deps) is a sensible future
> optimisation.

## Hosted demo — Streamlit Community Cloud

The hero tabs run on the free tier; the repo is pre-wired for it.

1. Point a new app at `dashboard/app.py` on the `main` branch. Community Cloud
   installs **`dashboard/requirements.txt`** — a slim, torch-free dependency set
   that sits next to the entrypoint and therefore takes precedence over the root
   `requirements.txt`. That is what keeps the build inside the free tier.
2. **Artifacts.** On first load `_ensure_artifact_or_stop()` runs a *targeted*
   `dvc pull` of just `models/model.pkl` + the three `data/splits/*.parquet`
   (~1.1 MB) — never `data/raw` or the big PyTorch artifacts.
   - **While the DagsHub repo is private:** add app secrets `DAGSHUB_USER` and
     `DAGSHUB_TOKEN` (a read-scoped token) so the pull authenticates.
   - **Once the repo is public** (the recruiter-handover step): the pull works
     **anonymously — delete the secrets**, nothing else changes.
3. Claim a subdomain (e.g. `behaviordna` → `https://behaviordna.streamlit.app`) so
   it matches the badge/link at the top of the README — or update that one line to
   whatever URL Streamlit assigns.
4. The **Overview / Player Profiles / Predict / Session Explorer** tabs work on
   CPU. The **📡 Live Session** tab needs the LSTM-AE (PyTorch) + streaming engine,
   excluded from the slim image, so on hosted it shows a "run locally" note.

(HuggingFace Spaces works the same way: Streamlit SDK, the slim requirements, a
targeted `dvc pull` on startup.)

## API quick check

```bash
curl localhost:8000/health
curl -X POST localhost:8000/predict/player \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","speed_mean":0.5, ...}'   # full 25-feature body
```

---

## Notes & caveats

- **Not built/tested in this dev env** (no Docker daemon here) — the Dockerfile,
  compose, and entrypoint are reviewed and standard; build on your Docker host
  (`docker compose build`) and they're ready.
- The dashboard's Live tab and the streaming API need PyTorch; on CPU-only hosts
  they run but slower.
- Production hardening (auth on `/stream`, rate limiting, a slim image, a pinned
  ONNX-faithful model — see [FINDINGS](FINDINGS.md)) is out of scope for this
  portfolio demo and noted in the roadmap.
