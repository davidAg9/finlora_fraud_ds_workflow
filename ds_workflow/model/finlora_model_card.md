# FinLora Fraud Detection — Model Card

I trained this model on past transactions and tested it on future ones, so the
scores below reflect how it behaves on data it has never seen.

**My pick:** Logistic | **I flag fraud at:** 0.80 | **mlflow run:** 9ec93a6b8d3541728685ac6b26c6c003

## How each model scored on my future holdout
| model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic | 0.9819 | 0.9652 |
| RandomForest | 0.9791 | 0.9625 |
| CatBoost | 0.9777 | 0.9579 |
| XGBoost | 0.9693 | 0.9517 |

I judge on PR-AUC because only ~9% of transactions are fraud — accuracy would
flatter a model that just says 'legit' every time.

## What drives the predictions (SHAP, mean |value|)
bool__very_new_account=1.083, bool__velocity_spike=0.900, bool__new_account=0.679, num_log__account_age_days=0.472, num_log__txn_velocity_1h=0.447, num_plain__corridor_risk=0.424, num_plain__device_trust_score=0.412, cat__home_country_uk=0.317, num_plain__chargeback_history_count=0.297, cat__kyc_tier_low=0.279

## How I serve it
- Python callers: load `models:/finlora-fraud-detector` from MLflow and predict on raw cleaned rows — the pipeline (my derived flag, dtypes, log1p, one-hot, impute, scale, Logistic) is baked into the artifact, nothing to reimplement.
- Axum API: fetches the ONNX copy (`onnx/model.onnx` in this same run) from MLflow at startup and serves 49 engineered features in model/finlora_feature_schema.json order; fraud at probability ≥ 0.80, human review between 0.50 and that.

## Why I trust it isn't cheating
I retrained without risk_score_internal + chargeback_history_count and the scores
barely moved (CatBoost PR-AUC 0.9605 vs 0.9589) — the signal is behavioural.
