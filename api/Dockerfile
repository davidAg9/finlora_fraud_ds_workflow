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

# trixie (not bookworm): the prebuilt onnxruntime static libs ort downloads
# were compiled against GCC 13+ headers (_M_replace_cold et al.) — bookworm's
# GCC-12 libstdc++ links nothing and the build dies in undefined references.
# Verified with nm before switching; the runtime stage must match for the
# same reason (dynamic libstdc++ at container boot).
FROM rust:1.90-slim-trixie AS builder
# build-essential (C++ linker for aws-lc, pulled in by reqwest's TLS),
# pkg-config + OpenSSL headers (reqwest links OpenSSL at compile time).
# Found both the hard way, one failed docker build at a time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists
WORKDIR /build
# Dependency layer first: rebuilding after a source edit reuses this cache.
COPY Cargo.toml Cargo.lock ./
# Stub the binary so `cargo build` fetches + compiles deps before sources exist.
RUN mkdir src && echo "fn main() {}" > src/main.rs \
    && cargo build --release \
    && rm -rf src
COPY src ./src
# touch: my sources are days older than the stub build's fingerprints, so
# without this cargo calls everything "fresh" and ships the STUB binary
# (ask me how I know — a 449KB server that exits in 48ms). Touching forces
# a real rebuild of the actual code.
RUN find src -type f -exec touch {} + && cargo build --release

FROM debian:trixie-slim
# ca-certificates: DagsHub/API calls are HTTPS. libssl3: reqwest's TLS at
# runtime. libgomp1: the prebuilt onnxruntime dylib ort downloads at compile
# time links against it — without this the binary dies on startup with a
# missing-library error.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libssl3 libgomp1 \
    && rm -rf /var/lib/apt/lists
COPY --from=builder /build/target/release/finlora-api /usr/local/bin/finlora-api
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/finlora-api"]
