# FinLora Fraud Detection — Model Card

I trained this model on past transactions and tested it on future ones, so the
scores below reflect how it behaves on data it has never seen.

**My pick:** RF-convertible (RandomForest, fully ONNX-bakeable) | **I flag fraud at:** 0.80 | **mlflow run:** 4144e624e86d433b9eb400c151c54578

## How each model scored on my future holdout
| model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic (best metrics) | 0.9819 | 0.9652 |
| RF-convertible (SHIPPED) | 0.9799 | 0.9628 |
| RandomForest (v8 setup) | 0.9791 | 0.9625 |
| CatBoost | 0.9775 | 0.9595 |
| XGBoost | 0.9693 | 0.9517 |

## Why I ship second place
Logistic wins on pure score (+0.0024 PR-AUC), but it needs
log1p — which no ONNX converter compiles — so serving it means maintaining
a parallel feature stack. The RF compiles raw->proba into ONE artifact:
no thresholds, layouts or log1p duplicated outside Python, ever. I measured
the price and took it; the @best alias records the decision, not just the winner.

## What drives the predictions (SHAP, mean |value|)
num__ip_risk_score=0.064, num__txn_velocity_1h=0.063, num__account_age_days=0.060, num__txn_velocity_24h=0.058, num__risk_score_internal=0.051, num__device_trust_score=0.035, num__amount_usd=0.017, cat__kyc_tier_low=0.016, num__fee=0.015, bool__location_mismatch=0.014

## How I serve it
- Python callers: load `models:/finlora-fraud-detector` from MLflow and predict
  on raw cleaned rows — the pipeline is baked into the artifact.
- Axum API: fetches the full raw->proba ONNX (`onnx/` in this run) from MLflow,
  takes raw transaction JSON, injects only the 2 ledger velocity counts, scores.
  Fraud at probability >= 0.80, human review between 0.50 and that.

## Why I trust it isn't cheating
Same temporal split and leakage checks as ever: the signal is behavioural.
