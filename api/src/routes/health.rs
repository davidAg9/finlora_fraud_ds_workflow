use axum::http::StatusCode;

// Simplest possible liveness probe — I hit this first whenever the demo
// misbehaves, to separate "server down" from "model unhappy".
pub async fn health_check() -> (StatusCode, &'static str) {
    (StatusCode::OK, "Finlora Fraud API is healthy")
}
