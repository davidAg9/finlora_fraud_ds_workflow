"""Export the winning FinLora model to ONNX for the axum/ort API.

Why ONNX at all: the axum service is Rust, and Rust can't run sklearn — it can
only run ONNX. So I split the pipeline in two: the Python-side feature math
(log1p, one-hot, missing-value handling) stays in Python (Streamlit/the baked
MLflow model), and only the trained estimator chain (imputer -> scaler ->
logistic) goes into the ONNX graph. The axum service receives the 49 engineered
features (frozen order in model/finlora_feature_schema.json) as a [1, 49] float
tensor. This script:

  1. retrains the winner from finlora_modeling.ipynb (same data, same temporal
     split, same feature contract via features.py),
  2. takes the fitted estimator steps,
  3. exports them to model/fl_fraud_model_v0.1.0 (fetched by the axum API from
     MLflow, which also stores a copy under onnx/ in the run),
  4. verifies sklearn vs onnxruntime parity on the test split (must agree 100%
     on fraud/not-fraud decisions, else I stop and investigate).

Run from ds_workflow/:  python export_onnx.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from features import ALL_FEATURE_COLS, derive_features, make_preprocessor, prepare_features

MODEL_PATH = os.path.join("model", "fl_fraud_model_v0.1.0")
TEST_THRESHOLD = 0.80

df = pd.read_parquet("data_assets/cleaned/finlora_cleaned.parquet")
df = df.sort_values("timestamp")
cut = int(len(df) * 0.8)

train = df.iloc[:cut]
test = df.iloc[cut:]

X_train = prepare_features(derive_features(train)[ALL_FEATURE_COLS])
y_train = train["is_fraud"]
X_test = prepare_features(derive_features(test)[ALL_FEATURE_COLS])
y_test = test["is_fraud"]

preprocessor, ALL_FEATURE_COLS = make_preprocessor()
preprocessor.fit(X_train)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())

best_pipe = Pipeline(
    [
        ("prep", preprocessor),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]
)
best_pipe.fit(X_train, y_train)

n_features = len(ALL_FEATURE_COLS)
print(f"raw input features: {n_features}")

# Trained estimator chain only — preprocessor stays in Python.
export_pipe = Pipeline(best_pipe.steps[1:])
Xe_test = preprocessor.transform(X_test).astype(np.float32)

n_features = Xe_test.shape[1]
if n_features != 49:
    raise SystemExit(f"engineered feature count mismatch: {n_features} != 49")
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