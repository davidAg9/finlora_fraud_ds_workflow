"""Parity lock: the API must match Python, row for row.

I run this after any retrain or API change. It takes raw test rows (plus nasty
edge cases: missing categories, unknown categories, missing amounts), scores
each two ways — the registry sklearn pipeline in Python vs POSTing raw JSON to
the API — and demands identical fraud/not-fraud decisions. If this fails, the
two sides have drifted and I fix it before serving.

Usage (API must be running, e.g. MODEL_URI=models:/finlora-fraud-detector@best):
    python parity_api.py                       # localhost:8000, version best
    API_URL=http://host:8000 python parity_api.py
"""

import json
import os
import urllib.request

import mlflow
import numpy as np
import pandas as pd

from features import RAW_FEATURE_COLS

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "best")
THRESHOLD = 0.80

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    tracking_uri = "sqlite:///" + os.path.abspath("mlruns/mlruns.db")
mlflow.set_tracking_uri(tracking_uri)


def post(payload):
    req = urllib.request.Request(
        API_URL + "/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def row_to_json(row):
    out = {}
    for c in RAW_FEATURE_COLS:
        v = row[c]
        out[c] = None if pd.isna(v) else (bool(v) if isinstance(v, (bool, np.bool_)) else v)
    # JSON can't do NaN cleanly and Rust treats null as unknown — same thing.
    for c in list(out):
        if isinstance(out[c], float) and np.isnan(out[c]):
            out[c] = None
    return out


def main():
    # "@best" is an alias (models:/name@alias); anything else is a version number.
    uri = (f"models:/finlora-fraud-detector{MODEL_VERSION}"
           if MODEL_VERSION.startswith("@")
           else f"models:/finlora-fraud-detector/{MODEL_VERSION}")
    model = mlflow.sklearn.load_model(uri)
    df = pd.read_parquet("data_assets/cleaned/finlora_cleaned.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * 0.8)
    te = df.iloc[cut:]

    # 25 diverse rows + 3 crafted edges: unknown category, missing category,
    # missing amount. No customer_id -> stateless path (vectorizer under test,
    # ledger uninvolved).
    sample = te.sample(25, random_state=7)
    edges = pd.DataFrame([{
        **{c: sample.iloc[0][c] for c in RAW_FEATURE_COLS},
        "channel": "pager",          # unseen category -> all-zero one-hot
    }, {
        **{c: sample.iloc[1][c] for c in RAW_FEATURE_COLS},
        "kyc_tier": None,            # missing category -> __missing__ bucket
    }, {
        **{c: sample.iloc[2][c] for c in RAW_FEATURE_COLS},
        "amount_usd": np.nan,        # missing amount -> median path
    }])
    rows = pd.concat([sample, edges], ignore_index=True)

    expected = model.predict_proba(rows[RAW_FEATURE_COLS])[:, 1]
    got, vels = [], []
    for _, row in rows.iterrows():
        r = post(row_to_json(row))
        got.append(r["probability"])
        vels.append((r["velocity_1h"], r["velocity_24h"]))
    got = np.array(got)
    assert all(v == (0, 0) for v in vels), "stateless rows must report zero velocity"

    max_diff = float(np.max(np.abs(expected - got)))
    agree = float(np.mean((expected >= THRESHOLD) == (got >= THRESHOLD)))
    print(f"rows: {len(rows)}  max |python - api| = {max_diff:.3e}  "
          f"agreement @ {THRESHOLD} = {agree:.4f}")
    if max_diff > 1e-4 or agree != 1.0:
        bad = np.argmax(np.abs(expected - got))
        print(f"WORST ROW {bad}: python={expected[bad]:.4f} api={got[bad]:.4f}")
        raise SystemExit("parity lock FAILED")
    print("parity lock OK — API matches Python")


if __name__ == "__main__":
    main()
