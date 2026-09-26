use axum::{extract::State, http::StatusCode, Json};
use ort::value::Tensor;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::model::schema::{
    ApiError, PredictionResponse, TransactionData, BOOL_COLS, CAT_COLS, NUM_COLS,
};
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

fn float_tensor(v: f32) -> Result<Tensor<f32>, String> {
    // Rank 2 ([1, 1]): the graph declares every input as [batch, 1].
    Tensor::from_array(([1_usize, 1_usize], vec![v].into_boxed_slice()))
        .map_err(|e| e.to_string())
}

fn string_tensor(s: String) -> Result<Tensor<String>, String> {
    Tensor::from_string_array(([1_usize, 1_usize], &*vec![s])).map_err(|e| e.to_string())
}

pub async fn prediction_handler(
    State(state): State<AppState>,
    Json(payload): Json<TransactionData>,
) -> Result<Json<PredictionResponse>, (StatusCode, Json<ApiError>)> {
    // Stateful path: a customer_id means "simulate". I count that customer's
    // stored clicks inside the 1h/24h windows and feed the COUNTS as the two
    // velocity inputs (the forest splits them directly — no thresholds, no
    // log scaling on my side). Only then do I record this click, so a
    // transaction never counts toward its own velocity (no lookahead).
    // Stateless path: no customer_id, no ledger — sent values pass through
    // (NaN flows to the ONNX median-imputer, "" to the __missing__ bucket,
    // exactly like training).
    let reference_ms = payload.timestamp_ms.unwrap_or_else(now_millis);
    let (ledger_counts, velocity_1h, velocity_24h) = match &payload.customer_id {
        Some(cid) => {
            let (v1, v24) = state.ledger.velocities(cid, reference_ms);
            state.ledger.record(cid, reference_ms);
            (Some((v1, v24)), v1, v24)
        }
        None => (None, 0, 0),
    };

    let num = |col: &str| -> f32 {
        match (col, ledger_counts) {
            ("txn_velocity_1h", Some((v1, _))) => v1 as f32,
            ("txn_velocity_24h", Some((_, v24))) => v24 as f32,
            _ => payload.num(col),
        }
    };

    let mut float_tensors = Vec::with_capacity(NUM_COLS.len() + BOOL_COLS.len());
    for col in NUM_COLS.iter().chain(BOOL_COLS.iter()) {
        let v = if BOOL_COLS.contains(col) {
            payload.flag(col)
        } else {
            num(col)
        };
        float_tensors.push(float_tensor(v).map_err(internal_error)?);
    }
    let mut string_tensors = Vec::with_capacity(CAT_COLS.len());
    for col in CAT_COLS.iter() {
        string_tensors.push(string_tensor(payload.cat(col)).map_err(internal_error)?);
    }

    let probability = {
        let mut session = state
            .model
            .lock()
            .map_err(|_| internal_error("session lock poisoned".to_string()))?;
        // Named inputs, resolved by the graph itself — order is irrelevant,
        // names must match export_onnx.py's initial_types.
        let mut ft = float_tensors.into_iter();
        let mut st = string_tensors.into_iter();
        let outputs = session
            .run(ort::inputs![
                "amount_usd" => ft.next().unwrap(),
                "fee" => ft.next().unwrap(),
                "account_age_days" => ft.next().unwrap(),
                "ip_risk_score" => ft.next().unwrap(),
                "device_trust_score" => ft.next().unwrap(),
                "chargeback_history_count" => ft.next().unwrap(),
                "risk_score_internal" => ft.next().unwrap(),
                "txn_velocity_1h" => ft.next().unwrap(),
                "txn_velocity_24h" => ft.next().unwrap(),
                "corridor_risk" => ft.next().unwrap(),
                "new_device" => ft.next().unwrap(),
                "location_mismatch" => ft.next().unwrap(),
                "corrupt_record" => ft.next().unwrap(),
                "timestamp_missing" => ft.next().unwrap(),
                "record_incomplete" => ft.next().unwrap(),
                "kyc_tier" => st.next().unwrap(),
                "channel" => st.next().unwrap(),
                "home_country" => st.next().unwrap(),
                "ip_country" => st.next().unwrap(),
                "source_currency" => st.next().unwrap(),
                "dest_currency" => st.next().unwrap()
            ])
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
