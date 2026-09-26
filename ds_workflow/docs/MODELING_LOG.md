# My Modelling Log — FinLora Fraud Detection

This is how I went from a cleaned dataset to a registered model, written the way
I'd explain it to someone. Numbers kept exact, jargon kept out.

My modelling notebook is `notebooks/finlora_modeling.ipynb`. The reusable feature
code lives in `features.py` so training and serving can never drift apart.

---

## How I split the data (and why it matters)

Fraud models must be trained on the **past** and tested on the **future** — that's
how they'd be used for real. So I sorted all 11,200 transactions by timestamp,
trained on the first 80% (8,960 rows) and tested on the last 20% (2,240 rows).
A random split would have leaked future patterns into training and flattered my
scores.

---

## The features I fed the models

I started from the raw cleaned columns and engineered 49 model inputs:

- **Skewed numbers** (amount, fee, account age, both velocities): I take `log1p`
  so a $5,000 transaction doesn't shout 100× louder than a $50 one.
- **Plain numbers** (risk scores, chargeback count, corridor risk): passed through.
- **Categories** (KYC tier, channel, countries, currencies): one-hot encoded, with an
  explicit `__missing__` bucket so the model can learn that "unknown" is itself a
  signal.
- **Flags** (new device, location mismatch, corrupt/incomplete records…): passed
  through as 0/1.
- **Six derived flags of my own**: `amount_high_flag` (amount ≥ $250 — catches 77%
  of all fraud at 2.4× baseline) plus five risk flags I added after testing the
  reference design's Step-4 idea against my data: `high_ip_risk`, `low_device_trust`,
  `new_account`, `very_new_account`, `velocity_spike`. Each cut is backed by a fraud-rate
  table in my EDA log, and together they lift PR-AUC by +0.002. I also tested time
  features (hour, weekend, late-night) and dropped them — they added nothing, so
  `timestamp` stays out of the serving contract.

One decision I want to be upfront about: tree models (Random Forest, XGBoost,
CatBoost) don't strictly *need* one-hot encoding, but I baked the **same**
  preprocessor into all four candidates anyway. That way every model lives in one
  shared feature space with one serving contract — a deliberate trade of a little
  elegance for a lot of operational sanity. (Since v11 the shipped RF uses the
  convertible variant of that preprocessor: no log1p, no custom steps.)

---

## I handled the imbalance with class weights, not resampling

Only ~9% of transactions are fraud. If I'd done nothing, every model would learn
that "always say legit" scores 91% and call it a day. So:

- Logistic Regression and Random Forest: `class_weight="balanced"`
- XGBoost: `scale_pos_weight = negatives / positives`
- CatBoost: `auto_class_weights="Balanced"`

I judged models on **PR-AUC** (how honest the ranking is when fraud is rare), not
accuracy.

---

## What I compared and what won

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| **Logistic Regression** | 0.9819 | **0.9652** |
| Random Forest | 0.9791 | 0.9625 |
| CatBoost | 0.9777 | 0.9579 |
| XGBoost | 0.9693 | 0.9517 |

Honestly, I expected a tree model to win — but plain logistic regression edged them
all on the temporal holdout. Simpler won, so I shipped simpler. The decision
threshold I use in serving is **0.80** (probabilities between 0.50 and 0.80 get
flagged for human review instead of auto-declined).

---

## I checked the model isn't cheating (leakage review)

Two features worried me because they correlate strongly with the label:
`risk_score_internal` and `chargeback_history_count`. Are they genuine behaviour or
is the answer leaking into the inputs? I retrained without them: performance barely
moved (CatBoost PR-AUC 0.9605 vs 0.9589). The signal is behavioural, not leaked —
but I'm keeping an eye on `risk_score_internal` in production.

I also verified the stored `txn_velocity_1h/24h` columns can't be reproduced from
transaction timestamps (0/1217 matches under every window definition I tried).
They're exogenous predictors, not something I can recompute — which is exactly why
the demo API computes velocity from its own simulation ledger instead.

---

## What SHAP told me (after I fixed my own bug)

My first SHAP run returned NaN for every feature. The bug was mine: I fed SHAP the
preprocessor output, but the pipeline imputes and scales *after* that step — so
SHAP saw NaN and garbage. I fixed it by feeding SHAP the exact input the estimator
receives. Corrected ranking (mean |SHAP|):

1. `account_age_days` — 0.670
2. **`txn_velocity_1h` — 0.465** (the 1-hour velocity leads the 24-hour one by ~14×)
3. `corridor_risk` — 0.449
4. `home_country_uk` — 0.388
5. `chargeback_history_count` — 0.385

---

## How I track and ship the model

- Every run is logged to **MLflow** (params, ROC/PR-AUC for all four models, the
  SHAP top-10, the model card). Registry name: `finlora-fraud-detector`.
- The full pipeline is baked into **one** logged artifact, so loading
  `models:/finlora-fraud-detector/<version>` gives you raw-row → probability with
  nothing to reimplement.
- Since v11 the champion is an **RF-convertible** pipeline (median/constant
  imputers, one-hot, forest — every op compiles) and the run carries a **full
  raw→proba ONNX** (`onnx/fl_fraud_model_v0.1.0`). The Axum API fetches it from
  MLflow at startup and takes **raw transaction JSON**: it forwards each field
  into its named ONNX input and injects only the 2 ledger velocity counts.
  No thresholds, no one-hot layout, no log1p anywhere outside Python — the
  `parity_api.py` lock (28 rows incl. missing/unknown edges, 100% agreement)
  proves the two sides match. One source of truth, two consumers.
- Locally I use a SQLite store (`ds_workflow/mlruns/mlruns.db`). For DagsHub I set:
  `MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow`,
  `MLFLOW_TRACKING_USERNAME=<dagshub-user>`,
  `MLFLOW_TRACKING_PASSWORD=<dagshub-token>` — then re-running the modelling
  notebook logs and registers straight into DagsHub (my registration cell uses
  the classic `runs:/` form on purpose, because DagsHub's server speaks the
  older MLflow protocol). I still need the DagsHub repo + a token for that push.
- The Axum API boots from `MODEL_URI` (`models:/<name>/<version>`,
  `runs:/<run>/<path>`, a URL, or a local file fallback) with HTTP Basic auth
  from the same `MLFLOW_TRACKING_USERNAME`/`PASSWORD` vars — exactly what
  DagsHub's MLflow endpoint speaks — so the API loads the model from DagsHub
  directly. MLflow is the single source of truth; no stale local model copies.

---

## Where this left me

A registered, explainable, leak-checked model (currently version 12,
RF-convertible behind the `@best` alias — Logistic still holds the pure-score
crown at 0.9652 vs 0.9628, and I took that −0.0024 trade deliberately for a
fully self-contained artifact). Served two ways from one MLflow run: raw-row →
probability for Python callers, full raw→proba ONNX for the Axum API — and a
Streamlit dashboard still to build on top.
