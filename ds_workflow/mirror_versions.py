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
    # Pin the destination artifact root explicitly. Default would be ./mlruns
    # relative to wherever I run this — which once polluted my real store, so
    # now I refuse to guess. For DagsHub this is ignored (server-side storage).
    artifact_root = os.environ.get("DEST_ARTIFACT_ROOT", "./mirror_artifacts")
    try:
        exp_id = dest.create_experiment("finlora-fraud", artifact_location=artifact_root)
    except Exception:
        exp_id = dest.get_experiment_by_name("finlora-fraud").experiment_id
    mlflow.set_experiment("finlora-fraud")

    for v in versions:
        src_ver = src.get_model_version(model_name, v)
        src_run = src.get_run(src_ver.run_id)
        print(f"mirroring {model_name} v{v} (run {src_ver.run_id}) ...")

        # I load the baked pipeline object itself (raw-row -> proba, no code
        # needed) and re-log it — the model content transfers bit-for-bit.
        mlflow.set_tracking_uri(src_uri)
        pipe = mlflow.sklearn.load_model(f"models:/{model_name}/{v}")
        mlflow.set_tracking_uri(dest_uri)

        with mlflow.start_run(run_name=f"mirror-{model_name}-v{v}") as run:
            mlflow.log_params(dict(src_run.data.params))
            mlflow.log_metrics({k: float(m) for k, m in src_run.data.metrics.items()})
            mlflow.sklearn.log_model(pipe, "model",
                                     serialization_format="cloudpickle")
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

        # I register through the fluent API (not create_model_version directly)
        # because only it resolves runs:/.../model to the MLflow-3 LoggedModel —
        # exactly what the modelling notebook does, so the result is identical.
        new_ver = mlflow.register_model(f"runs:/{dest_run_id}/model", model_name)
        dest.set_model_version_tag(
            model_name, new_ver.version, "mirrored_from_version", v)
        print(f"  -> {model_name} v{new_ver.version} "
              f"(was local v{v}, run {dest_run_id})")


if __name__ == "__main__":
    main()
