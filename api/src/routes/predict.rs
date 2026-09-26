use axum::{extract::State, http::StatusCode, Json};
use ort::value::Tensor;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::model::schema::{ApiError, PredictionResponse, TransactionPayload};
use crate::state::AppState;

// Cut-offs I share with the modelling notebook: auto-flag fraud at 0.80,
// and route the grey zone (0.50–0.80) to a human instead of auto-declining.
const FRAUD_THRESHOLD: f32 = 0.80;
const HUMAN_CONFIRM_LOWER_BOUND: f32 = 0.50;

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub async fn prediction_handler(
    State(state): State<AppState>,
    Json(payload): Json<TransactionPayload>,
) -> Result<Json<PredictionResponse>, (StatusCode, Json<ApiError>)> {
    let n = state.features.n_features;
    if payload.features.len() != n {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(format!(
                "expected {} features, got {}",
                n,
                payload.features.len()
            ))),
        ));
    }

    let mut features = payload.features;

    // Simulation velocity: when a customer_id is present the ledger owns every
    // velocity-derived slot. I count that customer's stored sim txns within the
    // last hour / last day, overwrite the client-sent placeholders, then record
    // the click so the next one sees an updated velocity.
    // Three slots, three encodings — all matching how Python engineered them:
    // - num_log__txn_velocity_1h/24h hold log1p(count), so I inject ln(1+v);
    // - bool__velocity_spike is the >=3/hr flag, so I inject 1.0/0.0 with the
    //   same cut the model trained on. The client can't compute this one: it
    //   only ever sees velocity 0 (its slots are placeholders), so if I didn't
    //   derive it here the model would never fire on bursts. My mistake in the
    //   first version — fixed.
    let reference_ms = payload.timestamp_ms.unwrap_or_else(now_millis);
    let (velocity_1h, velocity_24h) = match &payload.customer_id {
        Some(cid) => {
            let (v1, v24) = state.ledger.velocities(cid, reference_ms);
            features[state.features.velocity_1h] = (1.0 + v1 as f32).ln();
            features[state.features.velocity_24h] = (1.0 + v24 as f32).ln();
            features[state.features.velocity_spike] = if v1 >= 3 { 1.0 } else { 0.0 };
            state.ledger.record(cid, reference_ms);
            (v1, v24)
        }
        None => (0, 0),
    };

    let shape = [1_usize, n];
    let input_tensor = Tensor::from_array((shape, features.into_boxed_slice()))
        .map_err(|e| internal_error(e.to_string()))?;

    let probability = {
        let mut session = state
            .model
            .lock()
            .map_err(|_| internal_error("session lock poisoned".to_string()))?;
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| internal_error(e.to_string()))?;
        let output = outputs
            .get("probabilities")
            .ok_or_else(|| internal_error("model has no 'probabilities' output".to_string()))?;
        let (_, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| internal_error(e.to_string()))?;
        // probabilities = [p(legit), p(fraud)]; fraud probability is index 1
        data.get(1)
            .copied()
            .ok_or_else(|| internal_error("unexpected probabilities shape".to_string()))?
    };

    let is_fraud = probability >= FRAUD_THRESHOLD;
    let needs_human_confirmation = probability >= HUMAN_CONFIRM_LOWER_BOUND && !is_fraud;
    let confidence = (probability - HUMAN_CONFIRM_LOWER_BOUND).abs() * 2.0;
    let confidence = confidence.min(1.0);

    Ok(Json(PredictionResponse {
        confidence,
        probability,
        is_fraud,
        needs_human_confirmation,
        customer_id: payload.customer_id,
        velocity_1h,
        velocity_24h,
    }))
}

fn internal_error(message: String) -> (StatusCode, Json<ApiError>) {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiError::new(message)))
}