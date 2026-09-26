"""Mirror registered model versions from one MLflow tracking server to another.

I wrote this to move our last local versions to DagsHub. One honest limitation
up front: version *numbers* don't transfer — MLflow has no import API that
preserves them. So local v6..v9 become DagsHub v1..v4 in the same order, with the
same pipelines, params, metrics and ONNX artifacts. I keep a tag
(`mirrored_from_version`) on each new version so the lineage is never a mystery.

Usage (DagsHub):
    MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlruns/mlruns.db \\
    DEST_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow \\
    MLFLOW_TRACKING_USERNAME=<user> MLFLOW_TRACKING_PASSWORD=<token> \\
        python mirror_versions.py finlora-fraud-detector 6 7 8 9

Dry run (a scratch sqlite store, no cloud needed):
    MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlruns/mlruns.db" \\
    DEST_TRACKING_URI="sqlite:////tmp/mirror_test.db" \\
        python mirror_versions.py finlora-fraud-detector 6 7 8 9
"""

import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(2)
    model_name, versions = sys.argv[1], sys.argv[2:]
    src_uri = os.environ["MLFLOW_TRACKING_URI"]
    dest_uri = os.environ["DEST_TRACKING_URI"]

    mlflow.set_tracking_uri(src_uri)
    src = MlflowClient()
    mlflow.set_tracking_uri(dest_uri)
    dest = MlflowClient()
    # Artifact root: for LOCAL dest stores I pin it explicitly (default would be
    # ./mlruns relative to wherever I run this — which once polluted my real
    # store). For REMOTE servers (DagsHub) I must NOT send a local path — the
    # server decides storage, and a leaked path breaks every artifact silently.
    is_remote = dest_uri.startswith("http://") or dest_uri.startswith("https://")
    kwargs = {}
    if not is_remote:
        kwargs["artifact_location"] = os.environ.get("DEST_ARTIFACT_ROOT",
                                                     "./mirror_artifacts")
    # Destination experiment name is configurable: on DagsHub the natural
    # "finlora-fraud" name is squatted by a deleted experiment I poisoned
    # earlier (deleted names stay reserved), so there I use finlora-fraud-models.
    dest_exp = os.environ.get("DEST_EXPERIMENT", "finlora-fraud")
    try:
        exp_id = dest.create_experiment(dest_exp, **kwargs)
    except Exception:
        exp_id = dest.get_experiment_by_name(dest_exp).experiment_id
    # Guard against a poisoned experiment: if its artifact root is a local
    # path on a REMOTE server, every upload would vanish silently (this exact
    # failure cost me a full re-push once). Fail loud instead.
    if is_remote:
        loc = (dest.get_experiment(exp_id).artifact_location or "")
        if loc.startswith(("/tmp", "./", "/Users/", "/home/", "C:")) or loc == "":
            raise SystemExit(
                f"refusing to mirror: dest experiment artifact root is local "
                f"({loc!r}). Delete it server-side and re-run.")
    mlflow.set_experiment(dest_exp)

    for v in versions:
        src_ver = src.get_model_version(model_name, v)
        src_run = src.get_run(src_ver.run_id)
        print(f"mirroring {model_name} v{v} (run {src_ver.run_id}) ...")

        # I load the baked pipeline object itself (raw-row -> proba, no code
        # needed) and re-log it — the model content transfers bit-for-bit.
        mlflow.set_tracking_uri(src_uri)
        pipe = mlflow.sklearn.load_model(f"models:/{model_name}/{v}")
        mlflow.set_tracking_uri(dest_uri)

        import tempfile
        with mlflow.start_run(run_name=f"{model_name}-v{v}") as run:
            mlflow.log_params(dict(src_run.data.params))
            mlflow.log_metrics({k: float(m) for k, m in src_run.data.metrics.items()})
            # I save-then-log instead of log_model: the 3.x log_model speaks
            # LoggedModel protocol that older servers (DagsHub) accept yet
            # store nothing retrievable. Classic run-artifacts always land.
            with tempfile.TemporaryDirectory() as tmp:
                sig = None
                try:
                    import pandas as pd
                    from features import RAW_FEATURE_COLS
                    samp = pd.read_parquet(
                        "data_assets/cleaned/finlora_cleaned.parquet"
                    ).sort_values("timestamp")[RAW_FEATURE_COLS].head(50)
                    from mlflow.models import infer_signature
                    sig = infer_signature(samp, pipe.predict(samp))
                except Exception as e:
                    print(f"  signature skipped: {type(e).__name__}")
                mlflow.sklearn.save_model(pipe, os.path.join(tmp, "model"),
                                          signature=sig,
                                          serialization_format="cloudpickle")
                # log_artifactS (plural) uploads a directory's CONTENTS —
                # log_artifact would nest it as model/model/ (learned twice).
                dest.log_artifacts(run.info.run_id, os.path.join(tmp, "model"),
                                   artifact_path="model")
            # the ONNX twin the API boots from travels with the run too
            try:
                mlflow.set_tracking_uri(src_uri)
                onnx_file = mlflow.artifacts.download_artifacts(
                    run_id=src_ver.run_id,
                    artifact_path="onnx/fl_fraud_model_v0.1.0")
                mlflow.set_tracking_uri(dest_uri)
                mlflow.log_artifact(onnx_file, artifact_path="onnx")
            except Exception as e:
                print(f"  onnx copy skipped: {type(e).__name__}: {str(e)[:100]}")
            dest_run_id = run.info.run_id

        # I register through the classic create_model_version API with a runs:/
        # source, which every server generation understands (the fluent
        # register_model speaks MLflow-3 logged-model protocol and fails on
        # older servers like DagsHub's — learned the hard way).
        try:
            dest.create_registered_model(model_name)
        except Exception:
            pass  # already exists on re-runs
        new_ver = dest.create_model_version(
            model_name, f"runs:/{dest_run_id}/model", dest_run_id)
        dest.set_model_version_tag(
            model_name, new_ver.version, "mirrored_from_version", v)
        print(f"  -> {model_name} v{new_ver.version} "
              f"(was local v{v}, run {dest_run_id})")


if __name__ == "__main__":
    main()
