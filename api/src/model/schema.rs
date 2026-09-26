use serde::{Deserialize, Serialize};

// Shapes in and out of POST /predict. The vector layout comes from
// model/finlora_feature_schema.json (loaded at startup), so a Python retrain
// that adds features needs zero Rust changes — only the JSON moves.

#[derive(Debug, Deserialize)]
pub struct TransactionPayload {
    /// Engineered features in schema order. The velocity slots (log-scaled) are
    /// placeholders: when customer_id is present I overwrite them with the
    /// ledger's counts before scoring.
    pub features: Vec<f32>,
    /// Who is transacting. Present = demo mode (I compute velocity + record the
    /// click). Absent = plain scoring, nothing is stored.
    #[serde(default)]
    pub customer_id: Option<String>,
    /// Reference time in epoch millis. Optional: defaults to server now().
    /// Demo clicks advance the clock so velocity windows slide.
    #[serde(default)]
    pub timestamp_ms: Option<i64>,
}

/// Slot positions resolved from model/finlora_feature_schema.json at startup.
///
/// I used to hardcode these (3, 4, N=49). That meant every Python retrain that
/// added a feature needed a matching Rust edit — a silent-mismatch bug waiting
/// to happen. Now the JSON is the single source of truth and the API fails fast
/// at boot if a slot it owns is missing.
#[derive(Debug, Clone)]
pub struct FeatureIndices {
    pub n_features: usize,
    pub velocity_1h: usize,
    pub velocity_24h: usize,
    pub velocity_spike: usize,
}

impl FeatureIndices {
    const VELOCITY_1H_NAME: &'static str = "num_log__txn_velocity_1h";
    const VELOCITY_24H_NAME: &'static str = "num_log__txn_velocity_24h";
    const VELOCITY_SPIKE_NAME: &'static str = "bool__velocity_spike";

    pub fn load(path: &str) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("read schema {path}: {e}"))?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| format!("parse schema {path}: {e}"))?;
        let names = v
            .get("feature_names")
            .and_then(|n| n.as_array())
            .ok_or_else(|| format!("schema {path} has no feature_names list"))?;
        let find = |want: &str| {
            names
                .iter()
                .position(|n| n.as_str() == Some(want))
                .ok_or_else(|| format!("schema {path} is missing slot {want}"))
        };
        Ok(Self {
            n_features: names.len(),
            velocity_1h: find(Self::VELOCITY_1H_NAME)?,
            velocity_24h: find(Self::VELOCITY_24H_NAME)?,
            velocity_spike: find(Self::VELOCITY_SPIKE_NAME)?,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerDefaults {
    pub customer_id: String,
    pub kyc_tier: Option<String>,
    pub channel: Option<String>,
    pub home_country: Option<String>,
    pub ip_country: Option<String>,
    pub source_currency: Option<String>,
    pub dest_currency: Option<String>,
    pub account_age_days: Option<f32>,
    pub ip_risk_score: Option<f32>,
    pub device_trust_score: Option<f32>,
    pub chargeback_history_count: Option<f32>,
    pub risk_score_internal: Option<f32>,
    pub corridor_risk: Option<f32>,
    pub fee: Option<f32>,
    pub amount_usd: Option<f32>,
    pub new_device: Option<bool>,
    pub location_mismatch: Option<bool>,
    pub corrupt_record: Option<bool>,
    pub timestamp_missing: Option<bool>,
    pub record_incomplete: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct PredictionResponse {
    pub confidence: f32,
    pub probability: f32,
    pub is_fraud: bool,
    pub needs_human_confirmation: bool,
    pub customer_id: Option<String>,
    pub velocity_1h: u32,
    pub velocity_24h: u32,
}

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: String,
}

impl ApiError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            error: message.into(),
        }
    }
}