//! Rank-local inference probes. Scoring and numerical oracles stay in C++/CUDA.
use crate::{calibration::Environment, probes::ProbeSession, Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Protocol {
    #[serde(rename = "json-session-v1")]
    Session,
    #[serde(rename = "oneshot")]
    Oneshot,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolConfig {
    pub command: Vec<String>,
    pub inputs: PathBuf,
    pub capacity: u64,
    pub parents: u64,
    pub world: u32,
    pub environment: Environment,
    pub cwd: PathBuf,
    pub directory: PathBuf,
    pub protocol: Protocol,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankFailure {
    pub rank: u32,
    pub error: String,
    pub capacity: bool,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Batch {
    pub rows: Vec<Value>,
    pub failures: Vec<RankFailure>,
}
struct RankProcess {
    session: ProbeSession,
    log: PathBuf,
}
pub struct InferencePool {
    config: PoolConfig,
    ranks: Vec<Option<RankProcess>>,
    deadline: Instant,
    starts: u64,
    cancel: Arc<AtomicBool>,
    closed: bool,
}
pub fn cuda_capacity_rejection(text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    [
        "cuda out of memory",
        "cuda error: out of memory",
        "cudaerrormemoryallocation",
        "cublas_status_alloc_failed",
    ]
    .iter()
    .any(|s| text.contains(s))
}
pub fn validate_receipt(row: &Value, rank: u32, batch: u64, parents: u64) -> Result<()> {
    let seconds = row["seconds"]
        .as_array()
        .ok_or_else(|| Error::Message("missing inference timing cohort".into()))?;
    if row["batch"].as_u64() != Some(batch)
        || row["parents"].as_u64() != Some(parents)
        || row["device"].as_u64() != Some(rank as u64)
        || row["correctness_passed"] != true
        || row["numeric_error"].as_u64() != Some(0)
        || seconds.len() < 5
        || seconds
            .iter()
            .any(|v| !v.as_f64().is_some_and(|n| n.is_finite() && n > 0.))
        || row["torch_reserved_peak_bytes"].as_u64().is_none()
    {
        return Err(Error::Message("invalid inference session receipt".into()));
    }
    Ok(())
}
impl InferencePool {
    pub fn new(mut config: PoolConfig, timeout: Duration) -> Result<Self> {
        if config.command.is_empty()
            || config.capacity == 0
            || config.capacity > 65536
            || config.parents < config.capacity
            || config.parents > u32::MAX as u64
            || !(1..=65535).contains(&config.world)
            || timeout.is_zero()
        {
            return Err(Error::Message(
                "invalid inference calibration dimensions/timeout".into(),
            ));
        }
        config.cwd = config.cwd.canonicalize()?;
        config.inputs = config.inputs.canonicalize()?;
        if !config.directory.is_absolute() {
            config.directory = std::env::current_dir()?.join(&config.directory);
        }
        fs::create_dir(&config.directory)?;
        config.environment.remove("BEAM_ENSEMBLE_INFERENCE_MICRO");
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| Error::Message("inference deadline overflow".into()))?;
        let ranks = (0..config.world).map(|_| None).collect();
        Ok(Self {
            config,
            ranks,
            deadline,
            starts: 0,
            cancel: Arc::new(AtomicBool::new(false)),
            closed: false,
        })
    }
    pub fn cancellation(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }
    pub fn bind_cancellation(&mut self, cancel: Arc<AtomicBool>) -> Result<()> {
        if self.starts != 0 || self.closed {
            return Err(Error::Message(
                "cannot replace active inference cancellation".into(),
            ));
        }
        self.cancel = cancel;
        Ok(())
    }
    pub fn starts(&self) -> u64 {
        self.starts
    }
    pub fn pids(&self) -> Vec<Option<u32>> {
        self.ranks
            .iter()
            .map(|r| r.as_ref().map(|r| r.session.pid()))
            .collect()
    }
    pub fn capacity(&self) -> u64 {
        self.config.capacity
    }
    pub(crate) fn parents(&self) -> u64 {
        self.config.parents
    }
    pub(crate) fn world(&self) -> u32 {
        self.config.world
    }
    pub(crate) fn log_directory(&self) -> &std::path::Path {
        &self.config.directory
    }
    fn remaining(&self, deadline: Instant) -> Result<Duration> {
        if self.closed || self.cancel.load(Ordering::Acquire) {
            return Err(Error::Message(
                "inference calibration cancelled or closed".into(),
            ));
        }
        deadline
            .min(self.deadline)
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or_else(|| Error::Message("inference calibration timed out".into()))
    }
    fn clear(&mut self) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut error = None;
        for rank in self.ranks.iter_mut().flatten() {
            if let Err(e) = rank.session.begin_close() {
                error = Some(e);
            }
        }
        for rank in &mut self.ranks {
            if let Some(mut rank) = rank.take() {
                if let Err(e) = rank.session.finish_close(deadline) {
                    error = Some(e);
                }
            }
        }
        error.map_or(Ok(()), Err)
    }
    pub fn close(&mut self) -> Result<()> {
        self.closed = true;
        self.clear()
    }
    fn start(&mut self, batch: u64, deadline: Instant) -> Result<Vec<usize>> {
        let mut starting = Vec::new();
        for rank in 0..self.ranks.len() {
            if self.ranks[rank]
                .as_mut()
                .map(|r| r.session.exit_code())
                .transpose()?
                .flatten()
                .is_some()
            {
                self.ranks[rank].take();
            }
            if self.ranks[rank].is_some() {
                continue;
            }
            self.remaining(deadline)?;
            // The process belongs to the whole tuning phase. A short individual
            // measurement deadline must not poison later compatible requests.
            let remaining = self.remaining(self.deadline)?;
            self.starts = self
                .starts
                .checked_add(1)
                .ok_or_else(|| Error::Message("inference start counter overflow".into()))?;
            let log = self
                .config
                .directory
                .join(format!("session-{}-rank-{rank}.log", self.starts));
            let mut command = self.config.command.clone();
            let capacity = if self.config.protocol == Protocol::Session {
                self.config.capacity
            } else {
                batch
            };
            command.extend([
                self.config.inputs.to_string_lossy().into_owned(),
                capacity.to_string(),
                self.config.parents.to_string(),
                rank.to_string(),
            ]);
            if self.config.protocol == Protocol::Session {
                command.push("--session".into());
            }
            let mut session = ProbeSession::launch(
                &command,
                &self.config.cwd,
                &self.config.environment,
                &log,
                remaining,
            )?;
            session.bind_cancellation(self.cancel.clone());
            self.ranks[rank] = Some(RankProcess { session, log });
            starting.push(rank);
        }
        Ok(starting)
    }
    fn readiness(
        &mut self,
        starting: &[usize],
        batch: u64,
        deadline: Instant,
    ) -> Result<Option<Batch>> {
        let mut pending = starting.to_vec();
        while !pending.is_empty() {
            self.remaining(deadline)?;
            let mut complete = Vec::new();
            for rank in &pending {
                let process = self.ranks[*rank]
                    .as_mut()
                    .ok_or_else(|| Error::Message("missing inference rank".into()))?;
                match process.session.try_receive() {
                    Ok(Some(row))
                        if row["ready"] == true && row["device"].as_u64() == Some(*rank as u64) =>
                    {
                        complete.push(*rank)
                    }
                    Ok(Some(_)) => {
                        return Err(Error::Message(
                            "inference rank did not acknowledge readiness".into(),
                        ))
                    }
                    Ok(None) => {}
                    Err(error) => {
                        let text = fs::read_to_string(&process.log).unwrap_or_default();
                        if !cuda_capacity_rejection(&text) {
                            return Err(error);
                        }
                        let failure = RankFailure {
                            rank: *rank as u32,
                            error: format!("CUDA capacity rejected before readiness: {error}"),
                            capacity: true,
                        };
                        self.config.capacity = self.config.capacity.min(batch);
                        self.clear()?;
                        return Ok(Some(Batch {
                            rows: vec![],
                            failures: vec![failure],
                        }));
                    }
                }
            }
            pending.retain(|rank| !complete.contains(rank));
            if !pending.is_empty() {
                thread::sleep(Duration::from_millis(5));
            }
        }
        Ok(None)
    }
    fn failed_rank(&mut self, rank: usize, error: Error, allow_capacity: bool) -> RankFailure {
        let mut capacity = false;
        if let Some(mut process) = self.ranks[rank].take() {
            let _ = process.session.close();
            capacity = allow_capacity
                && cuda_capacity_rejection(&fs::read_to_string(process.log).unwrap_or_default());
        }
        RankFailure {
            rank: rank as u32,
            error: error.to_string(),
            capacity,
        }
    }
    pub fn measure(&mut self, batch: u64, parents: u64, timeout: Duration) -> Result<Batch> {
        if batch == 0
            || batch > self.config.capacity
            || parents < batch
            || parents > u32::MAX as u64
            || timeout.is_zero()
        {
            return Err(Error::Message(
                "invalid inference batch/parent count/timeout".into(),
            ));
        }
        if self.config.protocol == Protocol::Oneshot && parents != self.config.parents {
            return Err(Error::Message(
                "one-shot inference must preserve its parent workload".into(),
            ));
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| Error::Message("inference deadline overflow".into()))?
            .min(self.deadline);
        let result = (|| {
            self.remaining(deadline)?;
            let starting = self.start(batch, deadline)?;
            if self.config.protocol == Protocol::Session {
                if let Some(rejected) = self.readiness(&starting, batch, deadline)? {
                    return Ok(rejected);
                }
            }
            let mut failures = Vec::new();
            let mut rows = Vec::new();
            let mut pending = Vec::new();
            for rank in 0..self.ranks.len() {
                let process = self.ranks[rank]
                    .as_mut()
                    .ok_or_else(|| Error::Message("missing inference rank".into()))?;
                if self.config.protocol == Protocol::Session {
                    if let Err(error) = process
                        .session
                        .send(&json!({"batch":batch,"parents":parents}))
                    {
                        failures.push(self.failed_rank(rank, error, true));
                        continue;
                    }
                }
                pending.push(rank);
            }
            while !pending.is_empty() {
                self.remaining(deadline)?;
                let mut complete = Vec::new();
                for rank in &pending {
                    let process = self.ranks[*rank]
                        .as_mut()
                        .ok_or_else(|| Error::Message("missing inference rank".into()))?;
                    let value = process.session.try_receive();
                    match value {
                        Ok(Some(row)) => {
                            let mut accepted = validate_receipt(&row, *rank as u32, batch, parents);
                            // A quality/identity failure cannot become a memory
                            // bound because an unrelated log mentions CUDA OOM.
                            let allow_capacity = accepted.is_ok();
                            if self.config.protocol == Protocol::Oneshot {
                                let remaining = deadline.saturating_duration_since(Instant::now());
                                if let Err(error) = process.session.finish(remaining) {
                                    accepted = Err(error);
                                }
                            } else if process.session.exit_code()?.is_some_and(|code| code != 0) {
                                accepted = Err(Error::Message(
                                    "inference process exited after receipt".into(),
                                ));
                            }
                            match accepted {
                                Ok(()) => rows.push(row),
                                Err(error) => {
                                    failures.push(self.failed_rank(*rank, error, allow_capacity))
                                }
                            }
                            complete.push(*rank);
                        }
                        Ok(None) => {}
                        Err(error) => {
                            failures.push(self.failed_rank(*rank, error, true));
                            complete.push(*rank);
                        }
                    }
                }
                pending.retain(|rank| !complete.contains(rank));
                if !pending.is_empty() {
                    thread::sleep(Duration::from_millis(5));
                }
            }
            rows.sort_by_key(|r| r["device"].as_u64());
            if self.config.protocol == Protocol::Oneshot {
                self.clear()?;
            }
            Ok(Batch { rows, failures })
        })();
        if result.is_err() {
            let _ = self.clear();
        }
        result
    }
}
impl Drop for InferencePool {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
