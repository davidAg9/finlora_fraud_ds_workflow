use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};

use crate::model::schema::{ApiError, CustomerDefaults};
use crate::state::AppState;

/// List customers for the Streamlit dropdown (id + defaults as one row).
pub async fn list_customers(
    State(state): State<AppState>,
) -> Result<Json<Vec<CustomerDefaults>>, (StatusCode, Json<ApiError>)> {
    Ok(Json((*state.customers).clone()))
}

/// One customer's defaults by id.
pub async fn get_customer(
    State(state): State<AppState>,
    Path(customer_id): Path<String>,
) -> Result<Json<CustomerDefaults>, (StatusCode, Json<ApiError>)> {
    state
        .customers
        .iter()
        .find(|c| c.customer_id == customer_id)
        .cloned()
        .map(Json)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(format!("customer {customer_id} not found"))),
            )
        })
}