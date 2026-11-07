//! Compatible resource lifetime and independently validated search outcomes.
use crate::{calibration::Environment, probes::ProbeSession, Error, ResidentWorkers, Result};
use beam_core::search::{PermutationGraph, TerminalParser, TerminalResult};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionIdentity {
    pub graph: PermutationGraph,
    pub graph_hash: String,
    pub model_artifact_hash: String,
    pub model_backend: String,
    pub runtime_build: Value,
    pub devices: Vec<u32>,
    pub beam_width: u64,
    pub max_steps: u32,
    pub options: Value,
    pub cuda_visible_devices: Option<String>,
    pub cuda_device_order: Option<String>,
}
impl SessionIdentity {
    pub fn validate(&self) -> Result<()> {
        self.graph.validate()?;
        let hash = |s: &str| {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        };
        if !hash(&self.graph_hash)
            || !hash(&self.model_artifact_hash)
            || self.model_backend.is_empty()
            || !self.runtime_build.is_object()
            || !self.options.is_object()
            || self.runtime_build["binary_sha256"]
                .as_str()
                .is_none_or(|s| !hash(s))
            || self.beam_width == 0
            || self.devices.is_empty()
            || self.devices.len() > 65535
            || self
                .devices
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.devices.len()
        {
            return Err(Error::Message("invalid prepared session identity".into()));
        }
        let physical = beam_core::Domain::permutation(
            self.graph.center.len() as u64,
            self.graph.generators.len() as u32,
            16,
        )?
        .state_bytes()?;
        self.beam_width
            .div_ceil(self.devices.len() as u64)
            .checked_mul(physical)
            .ok_or(beam_core::ContractError::Overflow("session frontier"))?;
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Launch {
    pub command: Vec<String>,
    pub protocol_command: Vec<String>,
    pub environment: Environment,
    pub cwd: PathBuf,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Outcome {
    pub path: Option<Vec<u32>>,
    pub elapsed_seconds: f64,
    pub effective_beam_width: Option<u64>,
    pub metadata: Value,
}
pub fn parse_terminal(
    path: &Path,
    graph: &PermutationGraph,
    start: &[u8],
    max_steps: Option<u32>,
) -> Result<TerminalResult> {
    let mut parser = TerminalParser::new(graph, start, max_steps)?;
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = Vec::new();
    const MAX_LINE: u64 = 16 << 20;
    loop {
        line.clear();
        if reader
            .by_ref()
            .take(MAX_LINE + 1)
            .read_until(b'\n', &mut line)?
            == 0
        {
            break;
        }
        if line.len() as u64 > MAX_LINE {
            return Err(Error::Message("native terminal line exceeds 16 MiB".into()));
        }
        parser.line(&String::from_utf8_lossy(&line))?;
    }
    Ok(parser.finish()?)
}
pub struct SearchSession {
    workers: Option<ResidentWorkers>,
    identity: Option<SessionIdentity>,
    metadata: Option<Value>,
    cancellation: Arc<AtomicBool>,
    closed: bool,
    owner: u32,
}
impl Default for SearchSession {
    fn default() -> Self {
        Self::new()
    }
}
impl SearchSession {
    pub fn new() -> Self {
        Self {
            workers: None,
            identity: None,
            metadata: None,
            cancellation: Arc::new(AtomicBool::new(false)),
            closed: false,
            owner: std::process::id(),
        }
    }
    pub fn cancellation(&self) -> Arc<AtomicBool> {
        self.cancellation.clone()
    }
    pub fn pids(&self) -> Vec<u32> {
        self.workers
            .as_ref()
            .map_or_else(Vec::new, ResidentWorkers::pids)
    }
    pub fn is_closed(&self) -> bool {
        self.closed
    }
    fn check_owner(&self) -> Result<()> {
        if self.owner != std::process::id() {
            return Err(Error::Message(
                "native session cannot be shared across processes".into(),
            ));
        }
        Ok(())
    }
    fn release(&mut self) -> Result<()> {
        self.identity.take();
        self.metadata.take();
        if let Some(mut workers) = self.workers.take() {
            workers.close()?;
        }
        Ok(())
    }
    pub fn close(&mut self) -> Result<()> {
        self.check_owner()?;
        self.closed = true;
        self.cancellation.store(true, Ordering::Release);
        self.release()
    }
    pub fn run(
        &mut self,
        identity: &SessionIdentity,
        launch: &Launch,
        start: &[u8],
        run_dir: &Path,
        timeout: Duration,
        metadata: &Value,
    ) -> Result<Outcome> {
        self.check_owner()?;
        if self.closed || self.cancellation.load(Ordering::Acquire) {
            return Err(Error::Message("native session cancelled or closed".into()));
        }
        let result = self.run_inner(identity, launch, start, run_dir, timeout, metadata);
        if result.is_err() {
            let _ = self.release();
        }
        result
    }
    fn run_inner(
        &mut self,
        identity: &SessionIdentity,
        launch: &Launch,
        start: &[u8],
        run_dir: &Path,
        timeout: Duration,
        metadata: &Value,
    ) -> Result<Outcome> {
        let started = Instant::now();
        let deadline = started
            .checked_add(timeout)
            .ok_or_else(|| Error::Message("session deadline overflow".into()))?;
        let remaining = || {
            deadline
                .checked_duration_since(Instant::now())
                .filter(|d| !d.is_zero())
                .ok_or_else(|| Error::Message("native session timed out".into()))
        };
        identity.validate()?;
        identity.graph.validate_start(start)?;
        if !metadata.is_object() || timeout.is_zero() {
            return Err(Error::Message("invalid session metadata/timeout".into()));
        }
        if self.identity.as_ref() != Some(identity) {
            self.release()?;
        }
        if start == identity.graph.center || identity.max_steps == 0 {
            let solved = start == identity.graph.center;
            return Ok(Outcome {
                path: solved.then(Vec::new),
                elapsed_seconds: 0.,
                effective_beam_width: None,
                metadata: if solved {
                    json!({"already_solved":true,"replay_valid":true})
                } else {
                    json!({"budget_exhausted":true})
                },
            });
        }
        if run_dir.join("native-outcome.json").exists() {
            return Err(Error::Message(
                "native outcome already exists in run directory".into(),
            ));
        }
        let reused = self.workers.is_some();
        if !reused {
            if launch.command.is_empty() || launch.protocol_command.is_empty() {
                return Err(Error::Message(
                    "empty native launch/protocol command".into(),
                ));
            }
            let mut probe = ProbeSession::launch(
                &launch.protocol_command,
                &launch.cwd,
                &launch.environment,
                &run_dir.join("resident-protocol.log"),
                remaining()?.min(Duration::from_secs(15)),
            )?;
            probe.bind_cancellation(self.cancellation.clone());
            let reply = probe.receive(remaining()?.min(Duration::from_secs(15)))?;
            if reply != json!({"protocol":"resident-search-v1"}) {
                return Err(Error::Message("runner lacks resident-search-v1".into()));
            }
            probe.finish(remaining()?.min(Duration::from_secs(15)))?;
            remaining()?;
            let mut workers = ResidentWorkers::launch(
                &launch.command,
                identity.devices.len() as u32,
                &run_dir.join("resident-session"),
                &launch.cwd,
                &launch.environment,
            )?;
            workers.bind_parent_cancellation(self.cancellation.clone());
            self.workers = Some(workers);
            self.identity = Some(identity.clone());
        }
        let workers = self
            .workers
            .as_mut()
            .ok_or_else(|| Error::Message("missing native workers".into()))?;
        workers.submit(start, run_dir, remaining()?)?;
        let terminal = parse_terminal(
            &run_dir.join("native.log"),
            &identity.graph,
            start,
            Some(identity.max_steps),
        )?;
        if terminal.effective_beam_width != Some(identity.beam_width) {
            return Err(Error::Message(
                "resident effective beam differs from requested".into(),
            ));
        }
        let expected = if terminal.path.is_some() {
            1
        } else {
            identity.devices.len() as u64
        };
        if terminal.metadata["terminal_records"].as_u64() != Some(expected) {
            return Err(Error::Message(
                "resident search has unexpected terminal record count".into(),
            ));
        }
        if terminal.metadata["collected_solutions"]
            .as_array()
            .is_some_and(|rows| {
                rows.iter().any(|r| {
                    r["owner_rank"]
                        .as_u64()
                        .is_none_or(|n| n >= identity.devices.len() as u64)
                })
            })
        {
            return Err(Error::Message(
                "collection contains unknown owner rank".into(),
            ));
        }
        let mut output = if reused {
            self.metadata
                .clone()
                .ok_or_else(|| Error::Message("missing resident metadata".into()))?
        } else {
            metadata.clone()
        };
        let fields = output
            .as_object_mut()
            .ok_or_else(|| Error::Message("invalid resident metadata".into()))?;
        for (key, value) in terminal.metadata.as_object().unwrap() {
            fields.insert(key.clone(), value.clone());
        }
        fields.insert("graph_hash".into(), json!(identity.graph_hash));
        fields.insert(
            "model_artifact_hash".into(),
            json!(identity.model_artifact_hash),
        );
        fields.insert("devices".into(), json!(identity.devices));
        fields.insert("requested_beam_width".into(), json!(identity.beam_width));
        fields.insert("replay_valid".into(), json!(terminal.path.is_some()));
        fields.insert("log_path".into(), json!(run_dir.join("native.log")));
        fields.insert("persistent_session".into(),json!({"reused":reused,"worker_pids":workers.pids(),"request_count":workers.sequence()}));
        if reused {
            let profile = fields.entry("profile").or_insert_with(|| json!({}));
            let profile = profile
                .as_object_mut()
                .ok_or_else(|| Error::Message("invalid resident profile".into()))?;
            profile.remove("observed_saturated_depths");
            profile.insert("resident_profile_reused".into(), json!(true));
            profile.insert(
                "original_calibration_wall_seconds".into(),
                profile
                    .get("calibration_total_wall_seconds")
                    .cloned()
                    .unwrap_or(Value::Null),
            );
            profile.insert("calibration_total_wall_seconds".into(), json!(0.0));
        }
        if !reused {
            self.metadata = Some(output.clone());
        }
        let outcome = Outcome {
            path: terminal.path,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            effective_beam_width: terminal.effective_beam_width,
            metadata: output,
        };
        remaining()?;
        crate::atomic_json(&run_dir.join("native-outcome.json"), &outcome)?;
        Ok(outcome)
    }
}
impl Drop for SearchSession {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
