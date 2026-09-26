"""Export a registered FinLora model to ONNX for the axum/ort API.

Why this file exists (not just a notebook cell): exporting is a build step.
Docker/CI import `export_registered()` with no notebook involved — and it
converts the EXACT artifact I evaluated and registered, never a retrain.
The modelling notebook calls this same function, so notebook, script and CI
all produce byte-identical ONNX from the same registered version.

Since v11 the ONNX holds the WHOLE stateless pipeline (raw values in,
probability out): median/constant imputers, one-hot, forest. Nothing custom —
log1p and pandas code do not compile, which is why the serving model is a
tree pipeline. The only thing the API adds at request time is the 2 ledger
velocity counts; everything else is graph math.

Usage (from ds_workflow/):
    python export_onnx.py                    # exports version 11 (MODEL_VERSION)
    MODEL_VERSION=best python export_onnx.py  # or: follow the best alias
    MLFLOW_TRACKING_URI=... python export_onnx.py  # or: export from DagsHub
"""

import os

import mlflow
import numpy as np
import pandas as pd

from features import (FEATURE_COLS_NUM, FEATURE_COLS_BOOL, FEATURE_COLS_CAT,
                      RAW_FEATURE_COLS)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "fl_fraud_model_v0.1.0")
TEST_THRESHOLD = 0.80


def export_registered(model_name="finlora-fraud-detector", version="11",
                      model_path=MODEL_PATH, threshold=TEST_THRESHOLD,
                      tracking_uri=None):
    """Convert a registered pipeline's prep->estimator tail to ONNX.

    I slice from "prep" onward: the ("norm", ...) training-hygiene step is
    plain pandas for registry callers, and typed ONNX tensors never need it.
    Returns (model_path, max_diff, agreement) after a parity check that must
    pass 100% — anything less and I raise instead of writing a bad artifact.
    """
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    from sklearn.pipeline import Pipeline

    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        # default to the mlruns store next to THIS file (not the caller's CWD,
        # which is notebooks/ when the modelling notebook calls me)
        tracking_uri = ("sqlite:///" + os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mlruns", "mlruns.db"))
    mlflow.set_tracking_uri(tracking_uri)
    pipe = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
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
    onnx.save(onx, model_path)
    print(f"saved {model_path} ({os.path.getsize(model_path)/1e6:.1f} MB)")
    print("graph outputs:", [o.name for o in onx.graph.output])

    print("verifying parity vs sklearn ...")
    import onnxruntime as ort

    df = pd.read_parquet(os.path.join(HERE, "data_assets", "cleaned",
                                        "finlora_cleaned.parquet"))
    df = df.sort_values("timestamp")
    cut = int(len(df) * 0.8)
    smp = df.iloc[cut:][RAW_FEATURE_COLS]

    sess = ort.InferenceSession(model_path)
    feed = {c: smp[c].astype("float64").values.reshape(-1, 1).astype(np.float32)
            for c in float_cols}
    for c in FEATURE_COLS_CAT:
        s = smp[c].astype(object).where(smp[c].notna(), "")
        feed[c] = s.values.reshape(-1, 1).astype(object)

    proba_sk = pipe.predict_proba(smp)[:, 1]
    outs = sess.run(None, feed)
    names = [o.name for o in sess.get_outputs()]
    proba_ort = np.asarray(outs[names.index("probabilities")])[:, 1]

    max_diff = float(np.max(np.abs(proba_sk - proba_ort)))
    agree = float(np.mean((proba_sk >= threshold) == (proba_ort >= threshold)))
    print(f"max |sklearn - onnxruntime| proba diff = {max_diff:.3e}")
    print(f"decision agreement @ {threshold}   = {agree:.4f}")
    if max_diff > 1e-5 or agree != 1.0:
        raise SystemExit("parity check failed")
    print("ONNX export OK — artifact ready for axum/ort")
    return model_path, max_diff, agree


def main():
    version = os.environ.get("MODEL_VERSION", "11")
    export_registered(version=version)


if __name__ == "__main__":
    main()
