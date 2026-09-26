"""FinLora fraud demo dashboard (Streamlit).

I keep this app deliberately thin: it collects inputs, POSTs raw JSON to the
Axum API, and displays what comes back. All feature math lives in the model
(Python training) and the ONNX graph; all velocity state lives in the API's
in-memory ledger. If the API restarts, velocities reset — the dashboard shows
that honestly instead of hiding it.

Run from anywhere:  streamlit run app_streamlit/app.py
Needs: the API up (default http://localhost:8000), e.g.
    MODEL_URI="models:/finlora-fraud-detector@best" ./api/target/release/finlora-api
"""

import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent   # repo root (finlora-fraud/)
DS = ROOT / "ds_workflow"                        # drift.py + mlruns live here

st.set_page_config(page_title="FinLora Fraud Demo", layout="wide")
st.title("FinLora — fraud detection demo")

# ---------------------------------------------------------------- API plumbing
api_url = st.sidebar.text_input("API URL", "http://localhost:8000").rstrip("/")

def api_get(path):
    with urllib.request.urlopen(f"{api_url}{path}", timeout=10) as r:
        return json.load(r)

def api_post(path, payload):
    req = urllib.request.Request(
        f"{api_url}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

try:
    api_get("/health")
    st.sidebar.success("API reachable")
except Exception as e:  # noqa: BLE001 — I show the error, not the traceback
    st.sidebar.error(f"API unreachable: {e}")
    st.stop()

# ------------------------------------------------------- customer + defaults
@st.cache_data(show_spinner="Loading customers…")
def load_customers(url):
    with urllib.request.urlopen(f"{url}/customers", timeout=30) as r:
        return json.load(r)

customers = load_customers(api_url)
ids = [c["customer_id"] for c in customers]
cid = st.selectbox("Customer", ids, format_func=lambda i: f"{i[:8]}…")
defaults = next(c for c in customers if c["customer_id"] == cid)

def dflt(key, fallback):
    """A default that survives missing values (None -> fallback)."""
    v = defaults.get(key)
    return fallback if v is None else v

# ------------------------------------------------------------------- inputs
st.subheader("Transaction (defaults come from this customer's history)")
c1, c2, c3 = st.columns(3)
with c1:
    amount = st.number_input("amount_usd", 0.0, 100000.0,
                             float(dflt("amount_usd", 100.0)), 10.0)
    fee = st.number_input("fee", 0.0, 10000.0, float(dflt("fee", 5.0)), 1.0)
    age = st.number_input("account_age_days", 0.0, 5000.0,
                          float(dflt("account_age_days", 365.0)), 1.0)
with c2:
    # sliders for the scores an analyst would actually nudge
    ip_risk = st.slider("ip_risk_score", 0.0, 1.0, float(dflt("ip_risk_score", 0.1)))
    dev_trust = st.slider("device_trust_score", 0.0, 1.0,
                          float(dflt("device_trust_score", 0.8)))
    int_risk = st.slider("risk_score_internal", 0.0, 1.0,
                         float(dflt("risk_score_internal", 0.2)))
    corridor = st.slider("corridor_risk", 0.0, 1.0, float(dflt("corridor_risk", 0.0)))
with c3:
    cb = st.number_input("chargeback_history_count", 0, 50,
                         int(dflt("chargeback_history_count", 0)))
    src = st.selectbox("source_currency", ["USD", "CAD", "GBP"],
                       index=["USD", "CAD", "GBP"].index(dflt("source_currency", "USD")))
    dst = st.selectbox("dest_currency",
                       ["USD", "CAD", "GBP", "EUR", "CNY", "INR", "MXN", "NGN", "PHP"],
                       index=["USD", "CAD", "GBP", "EUR", "CNY", "INR", "MXN", "NGN", "PHP"].index(dflt("dest_currency", "USD")))
    chan = st.selectbox("channel", ["mobile", "web", "atm"],
                        index=["mobile", "web", "atm"].index(dflt("channel", "mobile")))

c4, c5 = st.columns(2)
with c4:
    kyc = st.selectbox("kyc_tier", ["standard", "enhanced", "low"],
                       index=["standard", "enhanced", "low"].index(dflt("kyc_tier", "standard")))
    home = st.text_input("home_country", dflt("home_country", "us"))
    ipc = st.text_input("ip_country", dflt("ip_country", "us"))
with c5:
    new_dev = st.checkbox("new_device", bool(dflt("new_device", False)))
    loc_mm = st.checkbox("location_mismatch", bool(dflt("location_mismatch", False)))
    corrupt = st.checkbox("corrupt_record", bool(dflt("corrupt_record", False)))
    ts_miss = st.checkbox("timestamp_missing", bool(dflt("timestamp_missing", False)))
    rec_inc = st.checkbox("record_incomplete", bool(dflt("record_incomplete", False)))

# ------------------------------------------------- simulated clock + transact
if "sim_ms" not in st.session_state:
    # one clock per dashboard session; every click moves it forward so the
    # API's 1h/24h windows slide like real time would
    st.session_state.sim_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
if "history" not in st.session_state:
    st.session_state.history = []

step_min = st.number_input("minutes per click (clock step)", 1, 60 * 24, 10)
st.caption(f"simulated time: {datetime.fromtimestamp(st.session_state.sim_ms/1000, timezone.utc):%Y-%m-%d %H:%M UTC} — advances {step_min} min per click")

if st.button("Make transaction", type="primary"):
    payload = {
        "customer_id": cid, "timestamp_ms": st.session_state.sim_ms,
        "amount_usd": amount, "fee": fee, "account_age_days": age,
        "ip_risk_score": ip_risk, "device_trust_score": dev_trust,
        "chargeback_history_count": cb, "risk_score_internal": int_risk,
        "txn_velocity_1h": None, "txn_velocity_24h": None,  # ledger owns these
        "corridor_risk": corridor,
        "new_device": new_dev, "location_mismatch": loc_mm,
        "corrupt_record": corrupt, "timestamp_missing": ts_miss,
        "record_incomplete": rec_inc,
        "kyc_tier": kyc, "channel": chan, "home_country": home,
        "ip_country": ipc, "source_currency": src, "dest_currency": dst,
    }
    try:
        r = api_post("/predict", payload)
        st.session_state.history.append({
            "time": datetime.fromtimestamp(st.session_state.sim_ms/1000, timezone.utc).strftime("%H:%M"),
            "amount": amount, "p": round(r["probability"], 4),
            "verdict": "FRAUD" if r["is_fraud"] else ("REVIEW" if r["needs_human_confirmation"] else "legit"),
            "v1h": r["velocity_1h"], "v24h": r["velocity_24h"],
        })
        st.session_state.sim_ms += step_min * 60_000
    except Exception as e:  # noqa: BLE001
        st.error(f"predict failed: {e}")

if st.session_state.history:
    last = st.session_state.history[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("verdict", last["verdict"], f"p={last['p']}")
    m2.metric("velocity_1h", last["v1h"])
    m3.metric("velocity_24h", last["v24h"])
    st.progress(min(last["p"], 1.0), text="fraud probability")
    st.dataframe(st.session_state.history, use_container_width=True)
    st.caption("velocities live in the API's memory — restart the API and they reset to 0.")

# ------------------------------------------------------------ drift monitor
st.divider()
st.subheader("Drift monitor")
st.caption("Runs ds_workflow/drift.py (PSI + KS vs the frozen training baseline). "
           "Python owns MLflow — Rust has no MLflow client, and drift needs "
           "pandas/scipy anyway — so the dashboard shells out instead of the API.")
if st.button("Run drift check"):
    with st.spinner("scoring 21 features against baseline…"):
        try:
            proc = subprocess.run(
                [sys.executable, "drift.py"], cwd=str(DS),
                capture_output=True, text=True, timeout=300)
            out = proc.stdout or proc.stderr
            st.code(out, language="text")
            if "significant" in out and "0 significant" not in out:
                st.warning("drift above threshold — see table")
            else:
                st.success("no significant drift")
        except Exception as e:  # noqa: BLE001
            st.error(f"drift run failed: {e}")
