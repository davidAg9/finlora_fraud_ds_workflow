# FinLora fraud API image. Build/run docs live in README.md ("Docker" section);
# this file stays build-only.
#
# Two deliberate choices, both learned the hard way:
# - trixie (not bookworm): ort's prebuilt C++ libs need GCC-13+ libstdc++
#   symbols at link AND load time, so builder and runtime must match.
# - non-root runtime: a container escape shouldn't hand over the host.

# syntax=docker/dockerfile:1
FROM rust:1.90-slim-trixie AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# BuildKit cache mounts (registry + compiled deps) survive across builds, so
# later builds recompile only our crate. This replaced the old stub-main +
# touch hack. Mounts vanish with the layer — hence the explicit copy out.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release && cp target/release/finlora-api /build/finlora-api

FROM debian:trixie-slim AS runner
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 libgomp1 \
    && rm -rf /var/lib/apt/lists \
    && groupadd -g 10001 appgroup \
    && useradd -u 10001 -g appgroup -s /bin/sh -m appuser \
    && mkdir -p /app && chown appuser:appgroup /app
COPY --from=builder /build/finlora-api /usr/local/bin/finlora-api
# No model, data, or baselines baked in — all arrive via env + mounts, so the
# same image serves any registry version without rebuilding.
USER appuser
WORKDIR /app
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/finlora-api"]
