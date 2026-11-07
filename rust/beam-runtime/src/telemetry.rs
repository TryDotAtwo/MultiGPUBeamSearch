//! Bounded GPU telemetry owned by a calibration phase. Missing data rejects it.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    process::{Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Row {
    pub index: u32,
    pub uuid: String,
    pub temperature: i32,
    pub throttled: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Receipt {
    pub samples: Vec<Vec<Row>>,
    pub errors: Vec<String>,
    pub interval_seconds: f64,
}
pub fn parse(text: &str) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for line in text.lines() {
        let fields: Vec<_> = line.split(',').map(str::trim).collect();
        if fields.len() != 6
            || fields[3..]
                .iter()
                .any(|s| !matches!(*s, "Active" | "Not Active"))
        {
            return Err(Error::Message(
                "GPU throttling telemetry unavailable".into(),
            ));
        }
        let index = fields[0]
            .parse()
            .map_err(|_| Error::Message("invalid GPU telemetry index".into()))?;
        let temperature = fields[2]
            .parse()
            .map_err(|_| Error::Message("invalid GPU telemetry temperature".into()))?;
        if fields[1].is_empty() || !seen.insert(index) {
            return Err(Error::Message("missing or duplicate GPU telemetry".into()));
        }
        rows.push(Row {
            index,
            uuid: fields[1].into(),
            temperature,
            throttled: fields[3..].contains(&"Active"),
        });
    }
    if rows.is_empty() {
        return Err(Error::Message("missing GPU telemetry".into()));
    }
    Ok(rows)
}
impl Receipt {
    pub fn verified(&self, uuids: &[String], indices: &[u32]) -> bool {
        if !self.errors.is_empty() || self.samples.len() < 2 || indices.is_empty() {
            return false;
        }
        let normalize = |s: &str| s.to_ascii_lowercase().trim_start_matches("gpu-").to_owned();
        let by_uuid = uuids.len() == indices.len()
            && uuids.iter().all(|s| !s.is_empty() && s != "unavailable");
        let expected: BTreeSet<_> = if by_uuid {
            uuids.iter().map(|s| normalize(s)).collect()
        } else {
            indices.iter().map(ToString::to_string).collect()
        };
        if expected.len() != indices.len() {
            return false;
        }
        self.samples.iter().all(|sample| {
            let observed: Vec<_> = sample
                .iter()
                .map(|r| {
                    (
                        if by_uuid {
                            normalize(&r.uuid)
                        } else {
                            r.index.to_string()
                        },
                        r,
                    )
                })
                .collect();
            if observed.iter().map(|r| &r.0).collect::<BTreeSet<_>>().len() != observed.len() {
                return false;
            }
            expected.iter().all(|id| {
                observed
                    .iter()
                    .any(|(name, row)| name == id && !row.throttled)
            })
        })
    }
}
fn query() -> Result<Vec<Row>> {
    let fields="index,uuid,temperature.gpu,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown";
    let mut child = Command::new("nvidia-smi")
        .arg(format!("--query-gpu={fields}"))
        .arg("--format=csv,noheader,nounits")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
            result => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(Error::Message(format!(
                    "GPU telemetry query timed out or failed: {result:?}"
                )));
            }
        }
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(Error::Message("GPU telemetry query failed".into()));
    }
    parse(&String::from_utf8_lossy(&output.stdout))
}
fn sample(receipt: &Arc<Mutex<Receipt>>) {
    let value = query();
    if let Ok(mut receipt) = receipt.lock() {
        match value {
            Ok(rows) => receipt.samples.push(rows),
            Err(error) => receipt.errors.push(error.to_string()),
        }
    }
}
pub struct Telemetry {
    receipt: Arc<Mutex<Receipt>>,
    stop: Option<mpsc::Sender<()>>,
    reader: Option<JoinHandle<()>>,
}
impl Telemetry {
    pub fn start() -> Result<Self> {
        let receipt = Arc::new(Mutex::new(Receipt {
            samples: Vec::new(),
            errors: Vec::new(),
            interval_seconds: 5.,
        }));
        sample(&receipt);
        let (stop, wait) = mpsc::channel();
        let worker_receipt = receipt.clone();
        let reader = thread::Builder::new()
            .name("beam-gpu-telemetry".into())
            .spawn(move || {
                while matches!(
                    wait.recv_timeout(Duration::from_secs(5)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ) {
                    sample(&worker_receipt);
                }
            })?;
        Ok(Self {
            receipt,
            stop: Some(stop),
            reader: Some(reader),
        })
    }
    fn close(&mut self) -> Result<()> {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(reader) = self.reader.take() {
            reader
                .join()
                .map_err(|_| Error::Message("GPU telemetry reader panicked".into()))?;
        }
        Ok(())
    }
    pub fn finish(mut self) -> Result<Receipt> {
        self.close()?;
        sample(&self.receipt);
        let result = self
            .receipt
            .lock()
            .map_err(|_| Error::Message("GPU telemetry mutex poisoned".into()))?
            .clone();
        Ok(result)
    }
}
impl Drop for Telemetry {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
