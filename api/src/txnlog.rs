use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::Mutex;

/// Append-only log of every scored request, one JSON object per line.
///
/// Why this exists: drift monitoring needs PRODUCTION distributions, and the
/// simulation ledger only keeps timestamps. So each /predict appends its raw
/// fields + the model's answer here; GET /drift reads this file back and
/// compares it against the frozen training baseline. Monitoring must never
/// break serving: a failed write is reported to stderr, never to the client.
pub struct TxnLogger {
    path: String,
    file: Mutex<BufWriter<File>>,
}

impl TxnLogger {
    pub fn open(path: &str) -> Result<Self, String> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("create log dir for {path}: {e}"))?;
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("open txn log {path}: {e}"))?;
        Ok(Self {
            path: path.to_string(),
            file: Mutex::new(BufWriter::new(file)),
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn log(&self, line: &str) -> Result<(), String> {
        let mut guard = self
            .file
            .lock()
            .map_err(|_| "txn log lock poisoned".to_string())?;
        writeln!(guard, "{line}").map_err(|e| format!("write txn log: {e}"))?;
        guard.flush().map_err(|e| format!("flush txn log: {e}"))?;
        Ok(())
    }
}
