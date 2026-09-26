# FinLora fraud API — reproducible serving image.
#
# What I bake in: the compiled release binary only (multi-stage, so the final
# image has no Rust toolchain). What I deliberately DON'T bake in: the model
# (fetched from MLflow at boot via MODEL_URI — stale baked models are how
# demos lie), customer defaults, or the drift baseline (mounted or env-set).
#
# Build (needs a docker daemon — I write Dockerfiles on machines without one):
#   docker build -t finlora-api ./api
# Run:
#   docker run -p 8000:8000 \
#     -e MODEL_URI="models:/finlora-fraud-detector@best" \
#     -e MLFLOW_TRACKING_URI="https://dagshub.com/<user>/<repo>.mlflow" \
#     -e MLFLOW_TRACKING_USERNAME="<user>" \
#     -e MLFLOW_TRACKING_PASSWORD="<token>" \
#     -e CUSTOMERS_PATH="/data/customers_defaults.json" \
#     -e DRIFT_BASELINE_PATH="/data/drift_baseline.json" \
#     -e TXN_LOG_PATH="/data/txn_log.jsonl" \
#     -v ./ds_workflow/data_assets:/data:ro \
#     -v ./ds_workflow/model:/model:ro \
#     finlora-api

FROM rust:1.90-slim-bookworm AS builder
WORKDIR /build
# Dependency layer first: rebuilding after a source edit reuses this cache.
COPY Cargo.toml Cargo.lock ./
# Stub the binary so `cargo build` fetches + compiles deps before sources exist.
RUN mkdir src && echo "fn main() {}" > src/main.rs \
    && cargo build --release \
    && rm -rf src
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
# ca-certificates: DagsHub/API calls are HTTPS. libgomp1: the prebuilt
# onnxruntime dylib ort downloads at compile time links against it — without
# this the binary dies on startup with a missing-library error.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists
COPY --from=builder /build/target/release/finlora-api /usr/local/bin/finlora-api
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/finlora-api"]
