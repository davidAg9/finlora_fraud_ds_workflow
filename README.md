# FinLora Fraud Detection

I built an end-to-end fraud classifier for FinLora's transaction stream: temporal-split modelling, an explainable registered model, a Rust API that simulates live traffic (including transaction velocity), a Streamlit demo dashboard, and PSI/KS drift monitoring. One line per piece, then how to run it.

## How I built it (my working logs)

- `ds_workflow/docs/EDA_LOG.md` — every cleaning decision with evidence (first-person, plain words)
- `ds_workflow/docs/MODELING_LOG.md` — features, bake-offs, kept/dropped decisions, serving design
- `ds_workflow/notebooks/` — cleaning, EDA, plumbing (style reference), modelling (trains → evaluates → registers)

## The model (MLflow: `finlora-fraud-detector`, alias `@best`)

| Version | What | PR-AUC |
|---|---|---|
| 10 (local) | Logistic, 44 feats — best pure score | 0.9652 |
| **15 (local) / 5 (DagsHub) — SHIPPED** | RF-convertible, full raw→proba ONNX | 0.9628 |

I ship second place on purpose: Logistic needs `log1p`, which no ONNX converter compiles. The forest compiles raw-values-in → probability-out into one 1.4MB artifact (parity vs sklearn 2.3e-07, 100% decision agreement), so nothing is duplicated outside Python. The −0.0024 trade is measured, documented, and reversible via the alias.

## Architecture

```
Streamlit (raw form inputs) ──POST /predict──▶ Axum API ──▶ ONNX (ort)
        │                                        │  ▲
        │ exploration, gauges          ledger ───┘  │ MODEL_URI
        ▼                                     MLflow/DagsHub
GET /drift ◀── txn_log.jsonl + frozen baseline (PSI per feature)
```

- **API** (`api/`, Rust/axum): takes raw transaction JSON, injects 1h/24h velocity from its in-memory simulation ledger, scores with ONNX fetched from MLflow at boot (`MODEL_URI=models:/…@best`), appends every request to `txn_log.jsonl`, serves `GET /drift` (PSI vs frozen baseline, trigger-free).
- **Dashboard** (`app_streamlit/`): customer dropdown from `/customers`, sliders for the honest knobs, a simulated clock per click, click history, drift button.
- **Monitoring** (`ds_workflow/drift.py`): PSI + KS per feature into MLflow, logging only — no automatic retraining (retraining on drift without labels is how you automate a feedback loop).

## Run it

```bash
# data (DVC) + Python env
cd ds_workflow && dvc pull            # needs DagsHub remote (see docs)
pixi install

# API (serves registry v15 via @best)
cd ../api && MODEL_URI="models:/finlora-fraud-detector@best" \
  MLFLOW_TRACKING_URI="sqlite:///$(pwd)/../ds_workflow/mlruns/mlruns.db" \
  ./target/release/finlora-api

# dashboard
cd ../ds_workflow && pixi run streamlit run ../app_streamlit/app.py
```

DagsHub hosts the mirror: experiments, `finlora-fraud-detector` v1–v5 (← local v9/10/11/12/15, `@best` → v5), ONNX twins, per-version cards, SHAP artifacts, P/R/F1 @0.80, drift runs, DVC data.

```bash
# Docker (API + ONNX; model + data mounted, never baked)
docker build -t finlora-api ./api
```
