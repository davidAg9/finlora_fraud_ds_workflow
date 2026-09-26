"""Drift monitor: PSI + KS per feature, logged to MLflow (no triggers yet).

Why this exists: chargeback labels take 30–90 days to arrive, so I can't watch
accuracy in production. What I CAN watch today is whether incoming data still
looks like training data — a shift in P(X) is my early warning that the model
is scoring a world it never learned.

What it does:
  1. Takes the train split as the frozen BASELINE (bins cut on baseline only —
     cutting bins on production data would hide the very shift I hunt).
  2. Takes the test split as a stand-in for production (same code will compare
     real API logs later; the comparison frame just swaps out).
  3. Per feature: PSI for everything; KS (statistic + p-value) for continuous
     numerics; missing-rate delta everywhere (a sudden gap in a column IS drift).
  4. Logs every metric plus the frozen baseline artifact to MLflow.

Reading the verdicts (industry rules I follow):
  PSI < 0.10 no shift | 0.10–0.25 moderate, investigate | >= 0.25 significant.
  KS: p < 0.05 AND D > 0.10 (p-values alone lie at high N — with 2240 rows even
  a D of 0.04 rejects, so I demand the effect size too).

Usage (from ds_workflow/):
    python drift.py
    MLFLOW_TRACKING_URI=... python drift.py   # log to DagsHub instead
"""

import json
import os

import mlflow
import numpy as np
import pandas as pd
from scipy import stats

from features import (FEATURE_COLS_NUM, FEATURE_COLS_CAT, FEATURE_COLS_BOOL,
                      RAW_FEATURE_COLS)

EPS = 1e-4
KS_P = 0.05
KS_D = 0.10


def _smooth(x):
    """Epsilon floor so empty bins don't divide by zero or take log(0)."""
    return np.where(x == 0, EPS, x)


def psi_cont(baseline, production, bins):
    """PSI over frozen baseline-cut bins (proportions, not counts)."""
    e = _smooth(np.histogram(baseline, bins=bins)[0] / len(baseline))
    a = _smooth(np.histogram(production, bins=bins)[0] / len(production))
    return float(np.sum((a - e) * np.log(a / e)))


def cut_bins(baseline, k=10):
    """Decile edges from BASELINE only; deduped for zero-inflated columns."""
    b = np.asarray(baseline, dtype=float)
    b = b[~np.isnan(b)]
    edges = np.unique(np.percentile(b, np.linspace(0, 100, k + 1)))
    if len(edges) < 2:
        return None
    edges[0] -= 1e-5
    edges[-1] += 1e-5
    return edges


def _is_continuous(s):
    return s.name in FEATURE_COLS_NUM and s.nunique() > 12


def midpoint_bins(baseline):
    """Bin edges for discrete numerics: midpoints between sorted baseline
    uniques, guarded by infinities. Values never sit on an edge (they're the
    integers/scores the edges fall between), so float32 vs float64 comparison
    in the API gives identical bins — no string keys needed anywhere."""
    u = sorted(float(x) for x in pd.Series(baseline).dropna().unique())
    if len(u) < 2:
        return None
    edges = [float("-inf")] + [(u[i] + u[i + 1]) / 2 for i in range(len(u) - 1)]
    edges.append(float("inf"))
    return np.array(edges)


def _cat_key(v):
    """Canonical category key, identical in Python and Rust: bools lowercase.
    (I match numpy bools too — `np.True_ is True` is False in Python, which
    silently skipped them the first time I wrote this.)"""
    if isinstance(v, (bool, np.bool_)):
        return "true" if bool(v) else "false"
    return v


def psi_cat(baseline, production):
    """PSI over category proportions (missing folded into __missing__)."""
    eb = baseline.map(_cat_key).fillna("__missing__").value_counts(normalize=True)
    ea = production.map(_cat_key).fillna("__missing__").value_counts(normalize=True)
    cats = set(eb.index) | set(ea.index)
    e = _smooth(np.array([eb.get(c, 0.0) for c in cats]))
    a = _smooth(np.array([ea.get(c, 0.0) for c in cats]))
    return float(np.sum((a - e) * np.log(a / e)))


def ks_check(baseline, production):
    """KS test with the high-N guard: needs p AND effect size D."""
    d, p = stats.ks_2samp(baseline, production)
    return float(d), float(p), bool(p < KS_P and d > KS_D)


def psi_verdict(v):
    if v >= 0.25:
        return "SIGNIFICANT"
    if v >= 0.10:
        return "moderate"
    return "stable"


def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = "sqlite:///" + os.path.abspath("mlruns/mlruns.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("finlora-fraud")

    df = pd.read_parquet("data_assets/cleaned/finlora_cleaned.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * 0.8)
    base, prod = df.iloc[:cut], df.iloc[cut:]
    print(f"baseline n={len(base)} (train), comparison n={len(prod)} (test-as-prod-proxy)")

    metrics, baseline_artifact = {}, {"bins": {}, "base_props": {},
                                             "cat_props": {}, "missing_rate": {},
                                             "n_baseline": len(base)}
    print(f"{'feature':<24} {'PSI':>7} {'verdict':<11} {'KS-D':>7} {'KS-p':>9}  miss-drift")
    for col in RAW_FEATURE_COLS:
        b, p = base[col], prod[col]
        mb, mp = float(b.isna().mean()), float(p.isna().mean())
        baseline_artifact["missing_rate"][col] = mb
        metrics[f"miss_base_{col}"] = mb
        metrics[f"miss_prod_{col}"] = mp
        miss_flag = "  <-- missing-rate moved" if abs(mp - mb) > 0.01 else ""
        if col in FEATURE_COLS_NUM:
            # numerics, binned two ways so the API can recompute with pure
            # float comparison (no fragile cross-language string keys):
            # continuous -> deciles; discrete (counts, corridor scores) ->
            # midpoints between baseline uniques.
            bv, pv = b.dropna().values, p.dropna().values
            if _is_continuous(b):
                bins = cut_bins(bv)
            else:
                bins = midpoint_bins(bv)
            v = psi_cont(bv, pv, bins) if bins is not None else 0.0
            d, pv_, _ = ks_check(bv, pv)
            # The API's /drift endpoint recomputes this PSI from the log, so I
            # store everything it needs: frozen edges AND baseline proportions
            # (deduped/zero-inflated bins are not uniform — never assume 0.1).
            baseline_artifact["bins"][col] = bins.tolist() if bins is not None else None
            if bins is not None:
                e = np.histogram(bv, bins=bins)[0] / len(bv)
                baseline_artifact["base_props"][col] = [round(float(x), 6) for x in e]
            else:
                baseline_artifact["base_props"][col] = None
            metrics[f"psi_{col}"] = v
            metrics[f"ks_D_{col}"] = d
            metrics[f"ks_p_{col}"] = pv_
            print(f"{col:<24} {v:>7.4f} {psi_verdict(v):<11} {d:>7.4f} {pv_:>9.2e}{miss_flag}")
        else:
            v = psi_cat(b.astype(object), p.astype(object))
            metrics[f"psi_{col}"] = v
            eb = b.map(_cat_key).fillna("__missing__").value_counts(normalize=True)
            baseline_artifact["cat_props"][col] = {
                str(k): round(float(x), 6) for k, x in eb.items()}
            print(f"{col:<24} {v:>7.4f} {psi_verdict(v):<11} {'—':>7} {'—':>9}{miss_flag}")

    n_mod = sum(1 for c in RAW_FEATURE_COLS if 0.10 <= metrics[f"psi_{c}"] < 0.25)
    n_sig = sum(1 for c in RAW_FEATURE_COLS if metrics[f"psi_{c}"] >= 0.25)
    metrics["psi_n_moderate"] = n_mod
    metrics["psi_n_significant"] = n_sig
    print(f"\nsummary: {n_mod} moderate, {n_sig} significant (no triggers wired — logging only)")

    def json_safe(x):
        # Infinities are valid floats but INVALID json — serde_json rightly
        # rejects them. I store null and the API reads it back as ∓infinity.
        if isinstance(x, float) and (x == float("inf") or x == float("-inf")):
            return None
        if isinstance(x, list):
            return [json_safe(v) for v in x]
        if isinstance(x, dict):
            return {k: json_safe(v) for k, v in x.items()}
        return x

    baseline_artifact = json_safe(baseline_artifact)

    with mlflow.start_run(run_name="finlora-drift-baseline") as run:
        mlflow.log_params({"baseline": "train split (80% temporal)",
                           "comparison": "test split as prod proxy (20%)",
                           "bins": "deciles cut on baseline, frozen in artifact"})
        mlflow.log_metrics(metrics)
        art_path = "model/drift_baseline.json"
        with open(art_path, "w") as f:
            json.dump(baseline_artifact, f, indent=2)
        mlflow.log_artifact(art_path)
        print("logged run:", run.info.run_id)


if __name__ == "__main__":
    main()
