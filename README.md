---
title: Plutchik ERC Dashboard
emoji: 🎭
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# Plutchik Emotion Recognition in Conversation (ERC)

This repo implements **32-class Plutchik** emotion recognition with **sarcasm** and **intensity**, conversation **arc** analysis, **Captum** explainability, and a **Human-in-the-Loop** correction path. The main application code lives under [`plutchik_erc_dashboard/`](plutchik_erc_dashboard/).

Upstream project: [https://github.com/Krsnapriya/aegis](https://github.com/Krsnapriya/aegis)

## Features

- **Analyst dashboard** (Streamlit): radar, arcs, comparative mode, batch CSV profiling, optional full-context `/explain`.
- **Inference API** (FastAPI): `/predict`, `/predict/batch`, `/predict/arc`, `/explain`, `/correct`, API key + rate limiting.
- **Annotation / HITL**: correction flow and CDA contrastive gate (see `docs/` and `data/processed/ERC/CDA_GATE_README.md`).
- **Docs**: hardened roadmap, Phase 7 evidence runbook, vision HTML in `html/plutchik_vision_dashboard_v2.1.html`.

## Quick start

```bash
git clone https://github.com/Krsnapriya/aegis.git
cd aegis
pip install -r requirements.txt
```

## Configuration

The application is configured via environment variables. For local development, you can create a `.env` file in the root of the repository. In production, these should be set directly in your deployment environment.

| Variable | Description | Default |
|---|---|---|
| `PLUTCHIK_API_URL` | The URL of the backend inference server. | `http://localhost:8000` |
| `PLUTCHIK_API_KEY` | The API key for accessing authenticated routes. | `None` |
| `DATABASE_URL` | The connection string for the PostgreSQL database. | `None` |
| `PLUTCHIK_CDA_JSONL` | Path to the contrastive pairs dataset for training. | `None` |
| `PLUTCHIK_CDA_MIN_PAIRS` | Minimum number of contrastive pairs to use. | `None` |

### Inference API

From the package directory (paths resolve to `core/advanced_engine.py` and local `models/` automatically):

```bash
cd plutchik_erc_dashboard
python -m uvicorn inference_server:app --host 0.0.0.0 --port 8000
```

Or from the repository root:

```bash
PYTHONPATH=. python -m uvicorn plutchik_erc_dashboard.inference_server:app --host 0.0.0.0 --port 8000
```

API base: `http://localhost:8000` (set `PLUTCHIK_API_KEY` for authenticated routes).

### Dashboard

```bash
cd plutchik_erc_dashboard
streamlit run app.py
```

### Training

```bash
cd plutchik_erc_dashboard
python train_v2.py
```

Optional CDA merge: set `PLUTCHIK_CDA_JSONL` and `PLUTCHIK_CDA_MIN_PAIRS` (see `docs/PHASE7_EVIDENCE_RUNBOOK.md`).

## Deployment (HF Spaces / Docker)

This space can run via Docker and Nginx so Streamlit and FastAPI share one port (see repo `Dockerfile` / `nginx.conf`).

- **Health** (behind proxy): `https://YOUR_SPACE_URL/api/health`
- **Predict**: `https://YOUR_SPACE_URL/api/predict`

## API surface (reference)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health and model readiness |
| `/predict` | POST | Single utterance + session context |
| `/predict/batch` | POST | Batch scoring |
| `/predict/arc` | POST | Turn-by-turn arc + inflection hints |
| `/explain` | POST | Prediction + Captum attributions (full context window) |
| `/correct` | POST | HITL correction |

## Contributing

Fork [aegis](https://github.com/Krsnapriya/aegis), branch, commit, and open a PR.

## License

See [LICENSE](LICENSE) if present in the repository.
