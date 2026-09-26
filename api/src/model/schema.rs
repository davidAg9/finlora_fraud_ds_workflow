use serde::{Deserialize, Serialize};

// What POST /predict accepts and returns.
//
// Streamlit sends RAW transaction fields — no vectors, no sklearn, no Python
// feature code on the client. The API forwards each field into its named ONNX
// input (floats incl. NaN, strings incl. "" for unknown) and only the 2 ledger
// velocity counts are ever computed server-side. If you can fill a form, you
// can call this endpoint.

/// Raw transaction as the dashboard sends it. Every field is optional: missing
/// numbers become NaN (the ONNX median-imputer fills them, exactly like
/// training), missing categories become "" (the __missing__ bucket),
/// missing flags count as false. txn_velocity_* are placeholders the ledger
/// overwrites whenever customer_id is present.
#[derive(Debug, Deserialize, Serialize)]
pub struct TransactionData {
    #[serde(default)]
    pub customer_id: Option<String>,
    /// Reference time in epoch millis (defaults to server now). Demo clicks
    /// advance the clock so velocity windows slide.
    #[serde(default)]
    pub timestamp_ms: Option<i64>,

    #[serde(default)]
    pub amount_usd: Option<f32>,
    #[serde(default)]
    pub fee: Option<f32>,
    #[serde(default)]
    pub account_age_days: Option<f32>,
    #[serde(default)]
    pub ip_risk_score: Option<f32>,
    #[serde(default)]
    pub device_trust_score: Option<f32>,
    #[serde(default)]
    pub chargeback_history_count: Option<f32>,
    #[serde(default)]
    pub risk_score_internal: Option<f32>,
    #[serde(default)]
    pub txn_velocity_1h: Option<f32>,
    #[serde(default)]
    pub txn_velocity_24h: Option<f32>,
    #[serde(default)]
    pub corridor_risk: Option<f32>,

    #[serde(default)]
    pub new_device: Option<bool>,
    #[serde(default)]
    pub location_mismatch: Option<bool>,
    #[serde(default)]
    pub corrupt_record: Option<bool>,
    #[serde(default)]
    pub timestamp_missing: Option<bool>,
    #[serde(default)]
    pub record_incomplete: Option<bool>,

    #[serde(default)]
    pub kyc_tier: Option<String>,
    #[serde(default)]
    pub channel: Option<String>,
    #[serde(default)]
    pub home_country: Option<String>,
    #[serde(default)]
    pub ip_country: Option<String>,
    #[serde(default)]
    pub source_currency: Option<String>,
    #[serde(default)]
    pub dest_currency: Option<String>,
}

/// Raw numeric columns, in the same grouping the model trained on. Names must
/// match the ONNX graph inputs (see export_onnx.py) — the graph resolves them
/// by name, so order here is free.
pub const NUM_COLS: [&str; 10] = [
    "amount_usd",
    "fee",
    "account_age_days",
    "ip_risk_score",
    "device_trust_score",
    "chargeback_history_count",
    "risk_score_internal",
    "txn_velocity_1h",
    "txn_velocity_24h",
    "corridor_risk",
];

/// Flag columns travel as 0.0/1.0 floats (training casts them the same way).
pub const BOOL_COLS: [&str; 5] = [
    "new_device",
    "location_mismatch",
    "corrupt_record",
    "timestamp_missing",
    "record_incomplete",
];

/// Category columns travel as strings, "" when unknown.
pub const CAT_COLS: [&str; 6] = [
    "kyc_tier",
    "channel",
    "home_country",
    "ip_country",
    "source_currency",
    "dest_currency",
];

impl TransactionData {
    /// Raw numeric field, NaN when unknown (matches training: NaN -> median).
    pub fn num(&self, col: &str) -> f32 {
        match col {
            "amount_usd" => self.amount_usd,
            "fee" => self.fee,
            "account_age_days" => self.account_age_days,
            "ip_risk_score" => self.ip_risk_score,
            "device_trust_score" => self.device_trust_score,
            "chargeback_history_count" => self.chargeback_history_count,
            "risk_score_internal" => self.risk_score_internal,
            "txn_velocity_1h" => self.txn_velocity_1h,
            "txn_velocity_24h" => self.txn_velocity_24h,
            "corridor_risk" => self.corridor_risk,
            _ => None,
        }
        .unwrap_or(f32::NAN)
    }

    /// Raw 0/1 flag, 0.0 when unknown.
    pub fn flag(&self, col: &str) -> f32 {
        let v = match col {
            "new_device" => self.new_device,
            "location_mismatch" => self.location_mismatch,
            "corrupt_record" => self.corrupt_record,
            "timestamp_missing" => self.timestamp_missing,
            "record_incomplete" => self.record_incomplete,
            _ => None,
        }
        .unwrap_or(false);
        if v {
            1.0
        } else {
            0.0
        }
    }

    /// Raw category, "" when unknown (matches training: "" -> __missing__).
    pub fn cat(&self, col: &str) -> String {
        match col {
            "kyc_tier" => self.kyc_tier.clone(),
            "channel" => self.channel.clone(),
            "home_country" => self.home_country.clone(),
            "ip_country" => self.ip_country.clone(),
            "source_currency" => self.source_currency.clone(),
            "dest_currency" => self.dest_currency.clone(),
            _ => None,
        }
        .unwrap_or_default()
    }
}

/// One row of data_assets/customers_defaults.json: a customer's typical values,
/// served by GET /customers so the dashboard can offer a dropdown + defaults.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_tx() -> TransactionData {
        TransactionData {
            customer_id: None,
            timestamp_ms: None,
            amount_usd: None,
            fee: None,
            account_age_days: None,
            ip_risk_score: None,
            device_trust_score: None,
            chargeback_history_count: None,
            risk_score_internal: None,
            txn_velocity_1h: None,
            txn_velocity_24h: None,
            corridor_risk: None,
            new_device: None,
            location_mismatch: None,
            corrupt_record: None,
            timestamp_missing: None,
            record_incomplete: None,
            kyc_tier: None,
            channel: None,
            home_country: None,
            ip_country: None,
            source_currency: None,
            dest_currency: None,
        }
    }

    #[test]
    fn missing_values_follow_training_conventions() {
        let tx = empty_tx();
        assert!(tx.num("amount_usd").is_nan()); // NaN -> ONNX median
        assert_eq!(tx.cat("channel"), ""); // "" -> __missing__ bucket
        assert_eq!(tx.flag("new_device"), 0.0);
    }

    #[test]
    fn column_lists_cover_all_model_inputs() {
        assert_eq!(NUM_COLS.len() + BOOL_COLS.len() + CAT_COLS.len(), 21);
    }
}
