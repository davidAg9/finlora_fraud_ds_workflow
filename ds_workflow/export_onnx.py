"""Export the registered FinLora model to ONNX for the axum/ort API.

Why this file exists separately (not a notebook cell): exporting is a build
step, not an experiment. Docker/CI runs `python export_onnx.py` with no
notebook involved — and it must convert the EXACT artifact I evaluated and
registered, not a retrain that merely hopes to match it.

Why ONNX at all: the axum service is Rust, and Rust can't run sklearn — only
ONNX. So I split the pipeline in two. The Python-side feature math (my derived
flags, dtype cleanup, log1p, one-hot) stays in Python and inside the MLflow
model; only the trained estimator chain (imputer -> scaler -> logistic) goes
into the ONNX graph. The axum service receives the 49 engineered features
(frozen order in model/finlora_feature_schema.json) as a [1, 49] float tensor.

Steps:
  1. load the baked pipeline from the registry (default: version 10),
  2. slice off everything up to and including the "prep" step,
  3. convert that estimator tail to model/fl_fraud_model_v0.1.0,
  4. verify sklearn vs onnxruntime parity on the test split (100% decision
     agreement required — anything less and I stop and investigate).

Usage (from ds_workflow/):
    python export_onnx.py                    # exports version 10 (MODEL_VERSION)
    MODEL_VERSION=best python export_onnx.py  # or: follow the best alias
    MLFLOW_TRACKING_URI=... python export_onnx.py  # or: export from DagsHub
"""

import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from features import RAW_FEATURE_COLS

MODEL_VERSION = os.environ.get("MODEL_VERSION", "10")
MODEL_PATH = os.path.join("model", "fl_fraud_model_v0.1.0")
TEST_THRESHOLD = 0.80
EXPECTED_FEATURES = 49

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    tracking_uri = "sqlite:///" + os.path.abspath("mlruns/mlruns.db")
mlflow.set_tracking_uri(tracking_uri)

# The exact artifact I evaluated — same run, same weights, no retrain.
pipe = mlflow.sklearn.load_model(f"models:/finlora-fraud-detector/{MODEL_VERSION}")
step_names = [name for name, _ in pipe.steps]
print(f"registered pipeline steps: {step_names}")
prep_at = step_names.index("prep")
export_pipe = Pipeline(pipe.steps[prep_at + 1:])
print(f"exporting estimator tail: {[n for n, _ in export_pipe.steps]}")

# Engineered test matrix, built with the pipeline's OWN fitted steps —
# so the parity check below compares onnxruntime against this exact model.
df = pd.read_parquet("data_assets/cleaned/finlora_cleaned.parquet")
df = df.sort_values("timestamp")
cut = int(len(df) * 0.8)
X_raw_test = df.iloc[cut:][RAW_FEATURE_COLS]
prep = pipe.named_steps["prep"]
Xe_test = prep.transform(pipe.named_steps["raw"].transform(X_raw_test)).astype(np.float32)

n_features = Xe_test.shape[1]
if n_features != EXPECTED_FEATURES:
    raise SystemExit(f"engineered feature count mismatch: {n_features} != {EXPECTED_FEATURES}")
print(f"engineered features: {n_features}")

print("converting to ONNX ...")
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_types = [("features", FloatTensorType([None, n_features]))]
options = {id(export_pipe): {"zipmap": False}}
onnx_model = convert_sklearn(export_pipe, initial_types=initial_types, options=options)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, MODEL_PATH)
print("saved", MODEL_PATH, f"({os.path.getsize(MODEL_PATH)} bytes)")

output_names = [o.name for o in onnx_model.graph.output]
input_names = [i.name for i in onnx_model.graph.input]
print("graph inputs :", input_names)
print("graph outputs:", output_names)

print("verifying parity vs sklearn ...")
import onnxruntime as ort

sess = ort.InferenceSession(MODEL_PATH)
proba_sk = export_pipe.predict_proba(Xe_test)[:, 1]
proba_ort = sess.run(["probabilities"], {"features": Xe_test})[0][:, 1]

max_diff = float(np.max(np.abs(proba_sk - proba_ort)))
agree = float(np.mean((proba_sk >= TEST_THRESHOLD) == (proba_ort >= TEST_THRESHOLD)))
print(f"max |sklearn - onnxruntime| proba diff = {max_diff:.3e}")
print(f"decision agreement @ {TEST_THRESHOLD}   = {agree:.4f}")
if max_diff > 1e-5:
    raise SystemExit("parity check failed")
print("ONNX export OK — artifact ready for axum/ort")
