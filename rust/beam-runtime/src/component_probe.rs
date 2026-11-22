//! One native planning owner and one telemetry lifetime per component trial.
use crate::{
    artifacts,
    calibration::Environment,
    component_tuning::{ComponentEvaluator, ComponentProbe},
    planning::PlanningSession,
    telemetry::Telemetry,
    Error, Result,
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
pub struct NativeComponents {
    planner: PlanningSession,
    beam: u64,
    directory: PathBuf,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
    trial: u64,
    closed: bool,
}
impl NativeComponents {
    pub fn launch(
        command: &[String],
        environment: &Environment,
        world: u32,
        beam: u64,
        cwd: &Path,
        directory: &Path,
        timeout: Duration,
        cancel: Arc<AtomicBool>,
    ) -> Result<Self> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| Error::Message("component deadline overflow".into()))?;
        fs::create_dir(directory)?;
        let planner = PlanningSession::launch_cancellable(
            command,
            environment,
            world,
            cwd,
            &directory.join("planning"),
            timeout,
            cancel.clone(),
        )?;
        if !planner.supports_components() {
            return Err(Error::Message(
                "native exact-capacity component protocol unsupported".into(),
            ));
        }
        artifacts::atomic_json(
            &directory.join("workers.json"),
            &json!({"pids":planner.pids(),"world":world}),
        )?;
        Ok(Self {
            planner,
            beam,
            directory: directory.to_path_buf(),
            deadline,
            cancel,
            trial: 0,
            closed: false,
        })
    }
    fn check(&self) -> Result<()> {
        if self.closed || self.cancel.load(Ordering::Acquire) {
            return Err(Error::Message(
                "component calibration cancelled or closed".into(),
            ));
        }
        if Instant::now() >= self.deadline {
            return Err(Error::Message("component calibration timed out".into()));
        }
        Ok(())
    }
}
impl ComponentEvaluator for NativeComponents {
    fn admit(&mut self, environment: &Environment) -> Result<Vec<Value>> {
        self.check()?;
        self.planner.admit(self.beam, environment)
    }
    fn measure(&mut self, environment: &Environment, plans: &[Value]) -> Result<ComponentProbe> {
        self.check()?;
        self.trial = self
            .trial
            .checked_add(1)
            .ok_or_else(|| Error::Message("component trial overflow".into()))?;
        let telemetry = Telemetry::start()?;
        let result = self
            .planner
            .measure_components(self.beam, environment)
            .map_err(|error| {
                crate::probes::rank_error(error, &self.directory.join("planning"), plans.len())
            });
        let receipt = telemetry.finish()?;
        artifacts::atomic_json(
            &self.directory.join(format!("trial-{}.json", self.trial)),
            &json!({
            "environment":environment,"plans":plans,"telemetry":receipt,
            "rank_measurements":result.as_ref().ok(),"error":result.as_ref().err().map(ToString::to_string)}),
        )?;
        self.check()?;
        Ok(ComponentProbe {
            rows: result?,
            telemetry: receipt,
        })
    }
    fn close(&mut self) -> Result<()> {
        self.closed = true;
        self.planner.close()?;
        artifacts::atomic_json(
            &self.directory.join("closed.json"),
            &json!({"pids":self.planner.pids(),"exit_codes":self.planner.exit_codes()?}),
        )
    }
}
impl Drop for NativeComponents {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.close();
        }
    }
}
