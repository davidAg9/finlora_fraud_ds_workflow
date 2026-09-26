use std::sync::{Arc, Mutex};

use ort::session::Session;

use crate::routes::drift::DriftBaseline;
use crate::ledger::SimulationLedger;
use crate::model::schema::CustomerDefaults;

// Everything every request handler needs, shared safely across threads.
// I wrap the ONNX session in a Mutex because ort sessions aren't thread-safe;
// the ledger and customer list are read-mostly, so Arc keeps them cheap to share.
#[derive(Clone)]
pub struct AppState {
    pub model: Arc<Mutex<Session>>,
    /// In-memory simulation ledger: counts each "make transaction" click and
    /// derives txn_velocity_1h/24h for that customer on the next click.
    pub ledger: Arc<SimulationLedger>,
    /// Loaded from data_assets/customers_defaults.json (defaults per customer
    /// for the Streamlit demo dropdown).
    pub customers: Arc<Vec<CustomerDefaults>>,
    /// Append-only log of scored requests (production distributions for drift).
    pub txn_log: Arc<crate::txnlog::TxnLogger>,
    /// Frozen training baseline for GET /drift. None when unconfigured — the
    /// API must boot and serve without it (monitoring is optional, scoring
    /// is not).
    pub drift_baseline: Option<Arc<DriftBaseline>>,
}