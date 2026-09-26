use serde::Deserialize;

/// How the ONNX bytes reached us (logged at startup so deploys are auditable).
pub struct ResolvedModel {
    pub bytes: Vec<u8>,
    pub description: String,
}

/// ONNX artifact path inside a modeling run (logged by finlora_modeling.ipynb).
const ONNX_ARTIFACT_PATH: &str = "onnx/fl_fraud_model_v0.1.0";

fn tracking_base() -> Result<String, String> {
    std::env::var("MLFLOW_TRACKING_URI")
        .map(|u| u.trim_end_matches('/').to_string())
        .map_err(|_| "MODEL_URI needs MLflow but MLFLOW_TRACKING_URI is not set".to_string())
}

// Auth for the tracking server. DagsHub wants HTTP Basic auth (username + token
// as password); a plain Bearer token also works if MLFLOW_TRACKING_TOKEN is set.
// I read both at request time (not startup) so a missing token only fails when
// a remote fetch is actually attempted — local-file boots stay credential-free.
fn basic_auth() -> Option<(String, String)> {
    let user = std::env::var("MLFLOW_TRACKING_USERNAME").ok().filter(|s| !s.is_empty())?;
    let pass = std::env::var("MLFLOW_TRACKING_PASSWORD").ok().filter(|s| !s.is_empty())?;
    Some((user, pass))
}

fn bearer() -> Option<String> {
    std::env::var("MLFLOW_TRACKING_TOKEN")
        .ok()
        .filter(|t| !t.is_empty())
}

fn apply_auth(req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    if let Some((user, pass)) = basic_auth() {
        req.basic_auth(user, Some(pass))
    } else if let Some(token) = bearer() {
        req.bearer_auth(token)
    } else {
        req
    }
}

#[derive(Debug, Deserialize)]
struct VersionEnvelope {
    model_version: ModelVersion,
}

#[derive(Debug, Deserialize)]
struct ModelVersion {
    version: String,
    source: String,
    run_id: String,
}

async fn get_json<T: for<'de> Deserialize<'de>>(client: &reqwest::Client, url: &str) -> Result<T, String> {
    let req = apply_auth(client.get(url));
    let resp = req.send().await.map_err(|e| format!("GET {url}: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        let head: String = body.chars().take(200).collect();
        return Err(format!("GET {url}: HTTP {status} {head}"));
    }
    resp.json::<T>()
        .await
        .map_err(|e| format!("GET {url}: bad JSON: {e}"))
}

/// Resolve `models:/<name>/<version|stage|alias>` (or `name@alias`) to its run id.
async fn resolve_registered(client: &reqwest::Client, base: &str, name: &str, ref_: &str) -> Result<(String, String), String> {
    // alias form: models:/name@alias
    if ref_.is_empty() {
        return Err("models:/ URI needs a version, stage or @alias".to_string());
    }
    if let Some(alias) = ref_.strip_prefix('@') {
        let url = format!("{base}/api/2.0/mlflow/registered-models/alias");
        let full = format!("{url}?name={name}&alias={alias}");
        let env: VersionEnvelope = get_json(client, &full).await?;
        return Ok((env.model_version.run_id, env.model_version.version));
    }
    // numeric version first
    if ref_.chars().all(|c| c.is_ascii_digit()) {
        let full = format!("{base}/api/2.0/mlflow/model-versions/get?name={name}&version={ref_}");
        let env: VersionEnvelope = get_json(client, &full).await?;
        return Ok((env.model_version.run_id, env.model_version.version));
    }
    // otherwise treat as stage ("Staging"/"Production") then as alias
    #[derive(Debug, Deserialize)]
    struct LatestEnvelope {
        model_versions: Option<Vec<ModelVersion>>,
    }
    let full = format!("{base}/api/2.0/mlflow/registered-models/get-latest-versions?name={name}&stages=[\"{ref_}\"]");
    if let Ok(env) = get_json::<LatestEnvelope>(client, &full).await {
        if let Some(v) = env.model_versions.and_then(|mut vv| vv.pop()) {
            return Ok((v.run_id, v.version));
        }
    }
    let full = format!("{base}/api/2.0/mlflow/registered-models/alias?name={name}&alias={ref_}");
    let env: VersionEnvelope = get_json(client, &full).await?;
    Ok((env.model_version.run_id, env.model_version.version))
}

/// Download one artifact file from the tracking server (OSS `get-artifact` API).
async fn download_artifact(client: &reqwest::Client, base: &str, run_id: &str, path: &str) -> Result<Vec<u8>, String> {
    let url = format!("{base}/get-artifact?path={path}&run_uuid={run_id}");
    let req = apply_auth(client.get(&url));
    let resp = req.send().await.map_err(|e| format!("artifact {path}: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        let head: String = body.chars().take(200).collect();
        return Err(format!("artifact {path}: HTTP {status} {head}"));
    }
    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| format!("artifact {path}: read body: {e}"))
}

/// Fetch the ONNX model bytes from wherever MLflow (or disk) holds them.
///
/// I added this so the API never serves a stale local copy: the modelling
/// notebook logs the ONNX next to the sklearn model in every run, and this
/// function pulls that exact artifact at startup (the run id is printed in the
/// boot log, so any prediction can be traced back to its training run).
///
/// Sources, in order:
/// 1. `MODEL_URI` unset -> `MODEL_PATH` local file (previous behaviour).
/// 2. `MODEL_URI` = local file path -> load it.
/// 3. `MODEL_URI` = `http(s)://...` -> download it.
/// 4. `MODEL_URI` = `runs:/<run_id>/<path>` -> fetch from the tracking server.
/// 5. `MODEL_URI` = `models:/<name>/<ver|stage|@alias>` -> resolve the version's
///    run, fetch `onnx/fl_fraud_model_v0.1.0` from it (logged by the notebook).
/// DagsHub note: registry + tracking calls below are plain MLflow REST with Basic
/// auth, which is exactly what DagsHub's MLflow endpoint speaks — so
/// MODEL_URI=models:/... works against DagsHub as well as a local server.
pub async fn fetch_model_bytes() -> Result<ResolvedModel, String> {
    let uri = std::env::var("MODEL_URI").unwrap_or_default();
    if uri.is_empty() {
        let path = std::env::var("MODEL_PATH")
            .unwrap_or_else(|_| "../ds_workflow/model/fl_fraud_model_v0.1.0".to_string());
        let bytes = std::fs::read(&path).map_err(|e| format!("read {path}: {e}"))?;
        return Ok(ResolvedModel {
            bytes,
            description: format!("local file {path}"),
        });
    }
    if uri.starts_with("http://") || uri.starts_with("https://") {
        let client = reqwest::Client::new();
        let bytes = download_url(&client, &uri).await?;
        return Ok(ResolvedModel {
            bytes,
            description: format!("url {uri}"),
        });
    }
    if !uri.starts_with("models:/") && !uri.starts_with("runs:/") {
        let bytes = std::fs::read(&uri).map_err(|e| format!("read {uri}: {e}"))?;
        return Ok(ResolvedModel {
            bytes,
            description: format!("local file {uri}"),
        });
    }

    let base = tracking_base()?;
    let client = reqwest::Client::new();
    let (run_id, artifact_path, label) = if let Some(rest) = uri.strip_prefix("runs:/") {
        let (run_id, path) = rest.split_once('/').ok_or_else(|| {
            "runs:/ URI must look like runs:/<run_id>/<artifact-path>".to_string()
        })?;
        (run_id.to_string(), path.to_string(), format!("run {run_id} path {path}"))
    } else {
        let rest = uri.strip_prefix("models:/").unwrap_or(&uri);
        // models:/name@alias  OR  models:/name/ref
        let (name, ref_) = if let Some((n, a)) = rest.split_once('@') {
            (n, format!("@{a}"))
        } else {
            rest.split_once('/').map(|(n, r)| (n, r.to_string())).ok_or_else(|| {
                "models:/ URI must look like models:/<name>/<version|stage|@alias>".to_string()
            })?
        };
        let (run_id, version) = resolve_registered(&client, &base, name, &ref_).await?;
        let label = format!("models:/{name}/{version} (run {run_id})");
        (run_id, ONNX_ARTIFACT_PATH.to_string(), label)
    };
    let bytes = download_artifact(&client, &base, &run_id, &artifact_path).await?;
    Ok(ResolvedModel {
        bytes,
        description: format!("mlflow {label} via {base}"),
    })
}

async fn download_url(client: &reqwest::Client, url: &str) -> Result<Vec<u8>, String> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("GET {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("GET {url}: HTTP {}", resp.status()));
    }
    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| format!("GET {url}: read body: {e}"))
}
