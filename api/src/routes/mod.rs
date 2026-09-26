pub mod constants;
pub mod customers;
pub mod drift;
pub mod health;
pub mod predict;

use axum::{
    routing::{get, post},
    Router,
};

use crate::state::AppState;

// All endpoints in one place so the demo client (Streamlit) has a single map:
// health for uptime checks, predict for scoring, customers for the dropdown.
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route(
            &constants::HEALTH_ROUTE,
            get(health::health_check),
        )
        .route(
            &constants::PREDICT_ROUTE,
            post(predict::prediction_handler),
        )
        .route(
            &constants::CUSTOMERS_ROUTE,
            get(customers::list_customers),
        )
        .route(
            &constants::CUSTOMER_ROUTE,
            get(customers::get_customer),
        )
        .route(&constants::DRIFT_ROUTE, get(drift::drift_report))
        .with_state(state)
}