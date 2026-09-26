"""Resolve a MODEL_URI to a local ONNX file using the official MLflow client.

I use this as a backup path: the Axum API already fetches models:/ and runs:/
URIs natively over MLflow REST, but if that ever fails against a hosted server,
this script (official client = correct auth + protocols) resolves the URI first
and Axum boots on the downloaded file.

DagsHub:
    MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow \\
    MLFLOW_TRACKING_USERNAME=<dagshub-user> MLFLOW_TRACKING_PASSWORD=<token> \\
        python fetch_model.py "models:/finlora-fraud-detector/9" ../api/model/

Local equivalent:
    MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlruns/mlruns.db" \\
        python fetch_model.py "models:/finlora-fraud-detector/9" ../api/model/
"""

import os
import sys

import mlflow


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    model_uri, dest_dir = sys.argv[1], sys.argv[2]
    os.makedirs(dest_dir, exist_ok=True)
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri, dst_path=dest_dir
    )
    print(f"resolved {model_uri} -> {local_path}")


if __name__ == "__main__":
    main()
