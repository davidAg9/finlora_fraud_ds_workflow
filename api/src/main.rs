mod ledger;
mod model;
mod model_source;
mod routes;
mod state;
mod txnlog;

use ort::session::Session;
use state::AppState;
use std::sync::{Arc, Mutex};

fn load_session(
    model_bytes: &[u8],
    description: &str,
) -> Result<Session, Box<dyn std::error::Error>> {
    println!("Loading model from {description} ({} bytes)", model_bytes.len());
    // ort can only commit a model from a file path, not from memory — so when the
    // bytes arrive over the network (MLflow) I stage them in the temp dir first.
    let staged = std::env::temp_dir().join("finlora_model.onnx");
    std::fs::write(&staged, model_bytes)?;
    Ok(Session::builder()?.commit_from_file(&staged)?)
}

fn load_customers() -> Result<Vec<model::schema::CustomerDefaults>, Box<dyn std::error::Error>> {
    let path =
        std::env::var("CUSTOMERS_PATH").unwrap_or_else(|_| "../ds_workflow/data_assets/customers_defaults.json".to_string());
    println!("Loading customer defaults from {path}");
    let raw = std::fs::read_to_string(&path)?;
    let customers: Vec<model::schema::CustomerDefaults> = serde_json::from_str(&raw)?;
    println!("Loaded {} customer defaults", customers.len());
    Ok(customers)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing Finlora Fraud API");

    // MODEL_URI (models:/..., runs:/..., http(s)://..., local path) wins;
    // without it we keep the previous MODEL_PATH local-file behaviour.
    let resolved = model_source::fetch_model_bytes()
        .await
        .map_err(|e| format!("model load failed: {e}"))?;
    let session = load_session(&resolved.bytes, &resolved.description)?;
    let customers = load_customers()?;

    let log_path =
        std::env::var("TXN_LOG_PATH").unwrap_or_else(|_| "txn_log.jsonl".to_string());
    let txn_log = Arc::new(
        txnlog::TxnLogger::open(&log_path).map_err(|e| format!("txn log: {e}"))?,
    );
    println!("Transaction log -> {log_path}");

    // Monitoring is optional: boot and serve even when no baseline exists.
    let baseline_path = std::env::var("DRIFT_BASELINE_PATH").unwrap_or_else(|_| {
        "../ds_workflow/model/drift_baseline.json".to_string()
    });
    let drift_baseline = match routes::drift::DriftBaseline::load(&baseline_path) {
        Ok(b) => {
            println!(
                "Drift baseline: {} train rows from {baseline_path}",
                b.n_baseline
            );
            Some(Arc::new(b))
        }
        Err(e) => {
            println!("Drift baseline unavailable ({e}) — GET /drift will 503");
            None
        }
    };

    let state = AppState {
        model: Arc::new(Mutex::new(session)),
        ledger: Arc::new(ledger::SimulationLedger::new()),
        customers: Arc::new(customers),
        txn_log,
        drift_baseline,
    };

    let app = routes::create_router(state);

    let bind_addr = "0.0.0.0:8000";
    println!("Server running on http://{}", bind_addr);

    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}