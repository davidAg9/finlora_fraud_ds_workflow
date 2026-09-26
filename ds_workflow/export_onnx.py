"""Export the registered FinLora model to ONNX for the axum/ort API.

Why this file exists separately (not a notebook cell): exporting is a build
step, not an experiment. Docker/CI runs `python export_onnx.py` with no
notebook involved — and it must convert the EXACT artifact I evaluated and
registered, not a retrain that merely hopes to match it.

Since v11 the ONNX holds the WHOLE stateless pipeline (raw values in,
probability out): median/constant imputers, one-hot, forest. Nothing custom —
I verified every op compiles (log1p and pandas code do not, which is why the
tree model carries no log features). The only thing the API adds at request
time is the 2 ledger velocity counts; everything else is graph math.

Steps:
  1. load the baked pipeline from the registry (default: version 11),
  2. convert everything from the "prep" step onward (skipping only the
     training-side pandas normalizer, which typed ONNX tensors don't need),
  3. save to model/fl_fraud_model_v0.1.0,
  4. verify sklearn vs onnxruntime parity on the test split (100% decision
     agreement required — anything less and I stop and investigate).

Usage (from ds_workflow/):
    python export_onnx.py                    # exports version 11 (MODEL_VERSION)
    MODEL_VERSION=best python export_onnx.py  # or: follow the best alias
    MLFLOW_TRACKING_URI=... python export_onnx.py  # or: export from DagsHub
"""

import os

import mlflow
import numpy as np
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.pipeline import Pipeline

from features import (FEATURE_COLS_NUM, FEATURE_COLS_BOOL, FEATURE_COLS_CAT,
                      RAW_FEATURE_COLS, normalize_for_serving)

MODEL_VERSION = os.environ.get("MODEL_VERSION", "11")
MODEL_PATH = os.path.join("model", "fl_fraud_model_v0.1.0")
TEST_THRESHOLD = 0.80

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    tracking_uri = "sqlite:///" + os.path.abspath("mlruns/mlruns.db")
mlflow.set_tracking_uri(tracking_uri)

# The exact artifact I evaluated — same run, same weights, no retrain.
pipe = mlflow.sklearn.load_model(f"models:/finlora-fraud-detector/{MODEL_VERSION}")
step_names = [name for name, _ in pipe.steps]
print(f"registered pipeline steps: {step_names}")
prep_at = step_names.index("prep")
export_pipe = Pipeline(pipe.steps[prep_at:])
print("exported steps: prep +", [n for n, _ in export_pipe.steps[1:]])

float_cols = FEATURE_COLS_NUM + FEATURE_COLS_BOOL
initial_types = ([(c, FloatTensorType([None, 1])) for c in float_cols] +
                 [(c, StringTensorType([None, 1])) for c in FEATURE_COLS_CAT])
onx = to_onnx(export_pipe, initial_types=initial_types,
              options={id(export_pipe): {"zipmap": False}})
import onnx
onnx.save(onx, MODEL_PATH)
print("saved", MODEL_PATH, f"({os.path.getsize(MODEL_PATH)/1e6:.1f} MB)")
print("graph inputs :", len(onx.graph.input), "named raw columns")
print("graph outputs:", [o.name for o in onx.graph.output])

print("verifying parity vs sklearn ...")
import onnxruntime as ort

df = pd.read_parquet("data_assets/cleaned/finlora_cleaned.parquet")
df = df.sort_values("timestamp")
cut = int(len(df) * 0.8)
smp = df.iloc[cut:][RAW_FEATURE_COLS]
Xn = normalize_for_serving(smp)

sess = ort.InferenceSession(MODEL_PATH)
feed = {c: smp[c].astype("float64").values.reshape(-1, 1).astype(np.float32)
        for c in float_cols}
for c in FEATURE_COLS_CAT:
    s = smp[c].astype(object).where(smp[c].notna(), "")
    feed[c] = s.values.reshape(-1, 1).astype(object)

proba_sk = pipe.predict_proba(Xn)[:, 1]
outs = sess.run(None, feed)
names = [o.name for o in sess.get_outputs()]
proba_ort = np.asarray(outs[names.index("probabilities")])[:, 1]

max_diff = float(np.max(np.abs(proba_sk - proba_ort)))
agree = float(np.mean((proba_sk >= TEST_THRESHOLD) == (proba_ort >= TEST_THRESHOLD)))
print(f"max |sklearn - onnxruntime| proba diff = {max_diff:.3e}")
print(f"decision agreement @ {TEST_THRESHOLD}   = {agree:.4f}")
if max_diff > 1e-5 or agree != 1.0:
    raise SystemExit("parity check failed")
print("ONNX export OK — artifact ready for axum/ort")
