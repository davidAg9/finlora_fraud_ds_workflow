use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::model::schema::{ApiError, BOOL_COLS, CAT_COLS, NUM_COLS};
use crate::state::AppState;

const EPS: f32 = 1e-4;
const MISS_DELTA: f32 = 0.01;

/// Frozen training baseline, written by ds_workflow/drift.py. The API only
/// recomputes PSI here (the industry trigger metric); the full KS analysis
/// stays in drift.py, which has scipy and the complete baseline sample.
#[derive(Debug, Clone, Deserialize)]
struct RawBaseline {
    n_baseline: u64,
    bins: HashMap<String, Option<Vec<Option<f32>>>>,
    base_props: HashMap<String, Option<Vec<f32>>>,
    cat_props: HashMap<String, HashMap<String, f32>>,
    missing_rate: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct DriftBaseline {
    pub n_baseline: u64,
    pub bins: HashMap<String, Vec<f32>>,
    pub base_props: HashMap<String, Option<Vec<f32>>>,
    pub cat_props: HashMap<String, HashMap<String, f32>>,
    pub missing_rate: HashMap<String, f32>,
}

impl DriftBaseline {
    pub fn load(path: &str) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("read drift baseline {path}: {e}"))?;
        let parsed: RawBaseline =
            serde_json::from_str(&raw).map_err(|e| format!("parse drift baseline {path}: {e}"))?;
        // drift.py stores unbounded edges as null (Infinity is not valid
        // JSON); the first null means -inf, any later null means +inf.
        let mut bins = HashMap::new();
        for (col, edges) in parsed.bins {
            let Some(edges) = edges else { continue };
            let mut out = Vec::with_capacity(edges.len());
            let mut first = true;
            for e in edges {
                match e {
                    Some(x) => out.push(x),
                    None if first => out.push(f32::NEG_INFINITY),
                    None => out.push(f32::INFINITY),
                }
                first = false;
            }
            bins.insert(col, out);
        }
        Ok(Self {
            n_baseline: parsed.n_baseline,
            bins,
            base_props: parsed.base_props,
            cat_props: parsed.cat_props,
            missing_rate: parsed.missing_rate,
        })
    }
}

fn psi(expected: &[f32], counts: &[u64], total: u64) -> f32 {
    let total = total.max(1) as f32;
    expected
        .iter()
        .zip(counts.iter())
        .map(|(e, c)| {
            let e = e.max(EPS);
            let a = (*c as f32 / total).max(EPS);
            (a - e) * (a / e).ln()
        })
        .sum()
}

fn verdict(psi: f32) -> &'static str {
    if psi >= 0.25 {
        "SIGNIFICANT"
    } else if psi >= 0.10 {
        "moderate"
    } else {
        "stable"
    }
}

/// np.histogram convention: [edge[i], edge[i+1]) left-closed, last bin closed.
fn bin_index(edges: &[f32], x: f32) -> Option<usize> {
    if edges.len() < 2 {
        return None;
    }
    for i in 0..edges.len() - 1 {
        let last = i == edges.len() - 2;
        if x >= edges[i] && (x < edges[i + 1] || (last && x <= edges[i + 1])) {
            return Some(i);
        }
    }
    None
}

#[derive(Debug, Serialize)]
pub struct DriftFeature {
    pub feature: String,
    pub psi: f32,
    pub verdict: String,
    pub n: u64,
    pub missing_drift: bool,
}

#[derive(Debug, Serialize)]
pub struct DriftResponse {
    pub n_logged: u64,
    pub n_baseline: u64,
    pub features: Vec<DriftFeature>,
    pub n_moderate: u32,
    pub n_significant: u32,
}

/// GET /drift — PSI per feature over the logged requests vs the frozen
/// baseline. Read-only and trigger-free by design: it reports, a human (or a
/// future cron job) decides. 503 when no baseline is configured.
pub async fn drift_report(
    State(state): State<AppState>,
) -> Result<Json<DriftResponse>, (StatusCode, Json<ApiError>)> {
    let base = state.drift_baseline.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiError::new(
                "drift baseline not configured (DRIFT_BASELINE_PATH)",
            )),
        )
    })?;

    // numeric values and category keys per column, straight from the log lines
    let mut nums: HashMap<String, Vec<f32>> = HashMap::new();
    let mut cats: HashMap<String, Vec<String>> = HashMap::new();
    let mut missing: HashMap<String, u64> = HashMap::new();
    let mut n_logged: u64 = 0;
    let mut skipped: u64 = 0;
    let content = std::fs::read_to_string(state.txn_log.path()).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::new(format!("read txn log: {e}"))),
        )
    })?;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        n_logged += 1;
        let get = |col: &str| v.get(col);
        for col in NUM_COLS.iter() {
            match get(col).and_then(|x| x.as_f64()).map(|x| x as f32) {
                Some(x) if x.is_finite() => nums.entry(col.to_string()).or_default().push(x),
                _ => *missing.entry(col.to_string()).or_default() += 1,
            }
        }
        for col in CAT_COLS.iter().chain(BOOL_COLS.iter()) {
            let key = match get(col) {
                None => "__missing__".to_string(),
                Some(x) if x.is_null() => "__missing__".to_string(),
                Some(x) if x.is_boolean() => {
                    if x.as_bool().unwrap_or(false) {
                        "true".to_string()
                    } else {
                        "false".to_string()
                    }
                }
                // (Numbers can't arrive here: TransactionData deserialisation
                // rejects mistyped fields with 422 before anything is logged.)
                Some(x) => x.as_str().unwrap_or("__missing__").to_string(),
            };
            if key == "__missing__" {
                *missing.entry(col.to_string()).or_default() += 1;
            }
            cats.entry(col.to_string()).or_default().push(key);
        }
    }
    let _ = skipped;

    let mut features = Vec::new();
    let mut n_moderate = 0u32;
    let mut n_significant = 0u32;
    // No observations for this comparison: nothing to compare, report clean.
    // An all-missing column (or an empty log) against eps-smoothing would
    // scream SIGNIFICANT on zero evidence — the missing_drift flag carries
    // that signal instead, honestly.
    let mut push = |feature: &str, psi: f32, n: u64, miss: bool| {
        // Empty log: nothing observed anywhere, report fully clean.
        // All-missing column: PSI is meaningless on zero evidence (the
        // missing_drift flag carries that signal instead).
        let (psi, miss) = if n_logged == 0 {
            (0.0, false)
        } else if n == 0 {
            (0.0, miss)
        } else {
            (psi, miss)
        };
        let v = verdict(psi);
        if v == "moderate" {
            n_moderate += 1;
        } else if v == "SIGNIFICANT" {
            n_significant += 1;
        }
        features.push(DriftFeature {
            feature: feature.to_string(),
            psi,
            verdict: v.to_string(),
            n,
            missing_drift: miss,
        });
    };

    for col in NUM_COLS.iter() {
        let (Some(edges), Some(props)) =
            (base.bins.get(*col), base.base_props.get(*col))
        else {
            continue;
        };
        let Some(props) = props else { continue };
        let vals = nums.get(*col).cloned().unwrap_or_default();
        let mut counts = vec![0u64; props.len()];
        for x in &vals {
            if let Some(i) = bin_index(edges, *x) {
                if i < counts.len() {
                    counts[i] += 1;
                }
            }
        }
        let miss = missing.get(*col).copied().unwrap_or(0);
        let total = vals.len() as u64 + miss;
        let miss_drift =
            (miss as f32 / total.max(1) as f32 - base.missing_rate.get(*col).copied().unwrap_or(0.0)).abs()
                > MISS_DELTA;
        push(col, psi(props, &counts, total), vals.len() as u64, miss_drift);
    }
    for col in CAT_COLS.iter().chain(BOOL_COLS.iter()) {
        let Some(bmap) = base.cat_props.get(*col) else {
            continue;
        };
        let vals = cats.get(*col).cloned().unwrap_or_default();
        // baseline keys plus any brand-new production categories (eps baseline)
        let keys: Vec<&String> = bmap.keys().collect();
        let mut seen_extra = false;
        for k in vals.iter() {
            if !bmap.contains_key(k) && !seen_extra {
                seen_extra = true;
            }
        }
        let mut expected = Vec::new();
        let mut counts = Vec::new();
        for k in keys.iter() {
            expected.push(*bmap.get(*k).unwrap_or(&0.0));
            counts.push(vals.iter().filter(|x| x.as_str() == k.as_str()).count() as u64);
        }
        if seen_extra {
            expected.push(0.0);
            counts.push(
                vals.iter()
                    .filter(|x| !bmap.contains_key(x.as_str()))
                    .count() as u64,
            );
        }
        let miss = missing.get(*col).copied().unwrap_or(0);
        let total = vals.len() as u64;
        let miss_drift =
            (miss as f32 / total.max(1) as f32 - base.missing_rate.get(*col).copied().unwrap_or(0.0)).abs()
                > MISS_DELTA;
        push(col, psi(&expected, &counts, total), total, miss_drift);
    }

    Ok(Json(DriftResponse {
        n_logged,
        n_baseline: base.n_baseline,
        features,
        n_moderate,
        n_significant,
    }))
}
