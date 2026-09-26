use std::collections::HashMap;
use std::sync::Mutex;

/// In-memory per-customer simulation ledger.
///
/// Streamlit drives the demo: each "make transaction" click posts a
/// customer_id + reference timestamp. We keep the customer's past simulated
/// transactions in memory and derive txn_velocity_1h/24h as a trailing count
/// over the [reference - window, reference) window, then record the new txn
/// so the next click sees the updated velocity. No persistence needed for a
/// demo, hence the name.
#[derive(Default)]
pub struct SimulationLedger {
    txns: Mutex<HashMap<String, Vec<i64>>>, // customer_id -> sim txn timestamps (ms)
}

pub const VELOCITY_1H_WINDOW_MS: i64 = 3_600_000;
pub const VELOCITY_24H_WINDOW_MS: i64 = 86_400_000;

impl SimulationLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count the customer's simulated transactions strictly within
    /// (reference - window, reference). The just-posted txn is NOT yet
    /// recorded at velocity time, so it never counts against itself.
    fn count_in_window(&self, customer: &str, reference_ms: i64, window_ms: i64) -> u32 {
        let guard = self.txns.lock().unwrap();
        let Some(times) = guard.get(customer) else {
            return 0;
        };
        let lo = reference_ms - window_ms;
        times
            .iter()
            .filter(|&&t| t > lo && t < reference_ms)
            .count() as u32
    }

    pub fn velocities(
        &self,
        customer: &str,
        reference_ms: i64,
    ) -> (u32, u32) {
        (
            self.count_in_window(customer, reference_ms, VELOCITY_1H_WINDOW_MS),
            self.count_in_window(customer, reference_ms, VELOCITY_24H_WINDOW_MS),
        )
    }

    pub fn record(&self, customer: &str, timestamp_ms: i64) {
        let mut guard = self.txns.lock().unwrap();
        guard
            .entry(customer.to_string())
            .or_default()
            .push(timestamp_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_only_within_window_and_before_reference() {
        let ledger = SimulationLedger::new();
        let ref_ms = 10_000_000;
        ledger.record("c1", ref_ms - 30 * 60 * 1000); // 30m before -> in 1h
        ledger.record("c1", ref_ms - 2 * 60 * 60 * 1000); // 2h  -> 24h only
        ledger.record("c1", ref_ms - 7 * 24 * 60 * 60 * 1000); // 7d  -> neither
        ledger.record("other", ref_ms - 1000); // somebody else

        let (h1, h24) = ledger.velocities("c1", ref_ms);
        assert_eq!(h1, 1);
        assert_eq!(h24, 2);

        let (h1, h24) = ledger.velocities("c1", ref_ms + VELOCITY_1H_WINDOW_MS);
        assert_eq!(h1, 0); // the 30m-old txn is now 1h30m back
        assert_eq!(h24, 2); // both earlier txns remain inside the day
    }

    #[test]
    fn different_customers_isolated() {
        let ledger = SimulationLedger::new();
        ledger.record("c1", 5);
        ledger.record("c2", 5);
        let (h1, _) = ledger.velocities("c1", 1_000_000);
        assert_eq!(h1, 1);
    }
}