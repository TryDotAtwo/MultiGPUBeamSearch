//! Native process ownership for the coarse frontier job.
use crate::{
    artifacts,
    calibration::{Environment, Profile},
    downstream_job,
    frontier_job::Request,
    inference_admission::{self, Candidate},
    inference_job, maximum_beam,
    pipeline::{Config, PipelineProbe},
    planning::PlanningSession,
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

pub(crate) struct Owner<'a> {
    request: &'a Request,
    pub root: PathBuf,
    pub world: u32,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
}
impl<'a> Owner<'a> {
    pub fn new(request: &'a Request, cancel: Arc<AtomicBool>) -> Result<Self> {
        let budget = Duration::try_from_secs_f64(request.inference.options.timeout_seconds)
            .map_err(|_| Error::Message("invalid frontier deadline".into()))?;
        let deadline = Instant::now()
            .checked_add(budget)
            .ok_or_else(|| Error::Message("frontier deadline overflow".into()))?;
        let root = request.inference.run_dir.join("frontier-control");
        fs::create_dir(&root)?;
        Ok(Self {
            request,
            root,
            world: request.inference.signature["world_size"].as_u64().unwrap() as u32,
            deadline,
            cancel,
        })
    }
    fn remaining(&self) -> Result<Duration> {
        if self.cancel.load(Ordering::Acquire) {
            return Err(Error::Message("frontier calibration cancelled".into()));
        }
        self.deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or_else(|| Error::Message("frontier calibration timed out".into()))
    }
    pub fn verify(&self) -> Result<()> {
        self.remaining()?;
        self.request.execution.runner.verify()?;
        self.request.inference.helper.verify()?;
        artifacts::verify_all(&self.request.inference.artifacts)?;
        artifacts::verify_all(&self.request.execution.fixtures)?;
        self.remaining()?;
        Ok(())
    }
    fn directory(&self, name: &str) -> Result<PathBuf> {
        let path = self.root.join(name);
        fs::create_dir(&path)?;
        Ok(path)
    }
    pub fn inference(&self, beam: u64, bootstrap: bool, name: &str) -> Result<Value> {
        let mut request = self.request.inference.clone();
        request.run_dir = self.directory(name)?;
        request.beam_width = beam;
        request.capacity_bootstrap = bootstrap;
        request.options.calibration_seconds = request
            .options
            .calibration_seconds
            .min(self.remaining()?.as_secs_f64());
        inference_job::run(&request, self.cancel.clone())
    }
    pub fn environment(&self, inference: &Value) -> Result<Environment> {
        let micro = inference["parent_batch"]
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or_else(|| Error::Message("missing inference microbatch".into()))?;
        let reserve = inference["reserve_bytes"]
            .as_u64()
            .ok_or_else(|| Error::Message("missing verified inference reserve".into()))?;
        let outer = micro
            .checked_mul(self.request.execution.row_factor)
            .ok_or_else(|| Error::Message("input row count overflow".into()))?;
        let mut env = self.request.inference.environment.clone();
        env.extend([
            ("BEAM_B_MICRO".into(), outer.to_string()),
            ("BEAM_ENSEMBLE_INFERENCE_MICRO".into(), micro.to_string()),
            ("BEAM_ENSEMBLE_RESERVE_BYTES".into(), reserve.to_string()),
        ]);
        Ok(env)
    }
    fn planning_budget(&self) -> Result<Duration> {
        Ok(Duration::from_secs_f64(
            (self.request.inference.options.calibration_pipeline_seconds * 0.25).min(60.),
        )
        .min(self.remaining()?))
    }
    fn planner(
        &self,
        env: &Environment,
        directory: &Path,
        timeout: Duration,
    ) -> Result<PlanningSession> {
        let command = vec![
            self.request
                .execution
                .runner
                .path
                .to_string_lossy()
                .into_owned(),
            self.request.execution.puzzle_id.to_string(),
            "1".into(),
            "1024".into(),
        ];
        let planner = PlanningSession::launch_cancellable(
            &command,
            env,
            self.world,
            &self.request.execution.cwd,
            &directory.join("ranks"),
            timeout,
            self.cancel.clone(),
        )?;
        artifacts::atomic_json(
            &directory.join("workers.json"),
            &json!({"pids":planner.pids()}),
        )?;
        Ok(planner)
    }
    fn close(&self, planner: &mut PlanningSession, directory: &Path) -> Result<()> {
        planner.close()?;
        artifacts::atomic_json(
            &directory.join("closed.json"),
            &json!({"pids":planner.pids(),"exit_codes":planner.exit_codes()?}),
        )
    }
    pub fn admit_fixed(&self, beam: u64, inference: &Value) -> Result<Value> {
        let directory = self.directory("inference-admission")?;
        let deadline = Instant::now() + self.planning_budget()?;
        let (selected, _) = inference_admission::select_admitted(
            inference,
            self.request.inference.options.calibration_max_batch,
            beam,
            |candidate: &Candidate| {
                self.remaining()?;
                let budget = deadline
                    .checked_duration_since(Instant::now())
                    .ok_or_else(|| Error::Message("inference admission timed out".into()))?;
                let env =
                    self.environment(&inference_admission::with_candidate(inference, candidate))?;
                let path = directory.join(candidate.parent_batch.to_string());
                fs::create_dir(&path)?;
                let mut planner = self.planner(&env, &path, budget)?;
                let overrides = Environment::from([
                    ("BEAM_B_MICRO".into(), env["BEAM_B_MICRO"].clone()),
                    (
                        "BEAM_ENSEMBLE_INFERENCE_MICRO".into(),
                        candidate.parent_batch.to_string(),
                    ),
                ]);
                let result = planner.admit(beam, &overrides);
                let cleanup = self.close(&mut planner, &path);
                artifacts::atomic_json(
                    &path.join("admission.json"),
                    &json!({"candidate":candidate,"beam":beam,
                    "plans":result.as_ref().ok(),"error":result.as_ref().err().map(ToString::to_string)}),
                )?;
                let plans = result?;
                cleanup?;
                Ok(plans)
            },
        )?;
        artifacts::atomic_json(&directory.join("selection.json"), &selected)?;
        Ok(selected)
    }
    pub fn discover(&self, inference: &Value, working: bool, name: &str) -> Result<Value> {
        let directory = self.directory(name)?;
        let env = self.environment(inference)?;
        let parse = |key: &str, default: u64| -> Result<u64> {
            match env.get(key) {
                Some(s) => s
                    .parse::<u64>()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| Error::Message(format!("invalid {key}"))),
                None => Ok(default),
            }
        };
        let alignment = parse("BEAM_STREAM4_BATCH_ALIGNMENT", 1024)?;
        let ceiling = self.request.capacity_ceiling()?;
        let config = maximum_beam::Config {
            upper_bound: ceiling.upper_bound,
            alignment,
            world: self.world,
            outer: parse("BEAM_B_MICRO", 1)?,
            minimum_flush: parse("BEAM_STREAM4_BATCH_CANDIDATES", alignment.saturating_mul(8))?,
        };
        let budget = self.planning_budget()?;
        let deadline = Instant::now() + budget;
        let mut planner = self.planner(&env, &directory, budget)?;
        let result = maximum_beam::discover(&config, working, deadline, |width, profile| {
            self.remaining()?;
            planner.admit(width, profile)
        });
        let cleanup = self.close(&mut planner, &directory);
        let mut data = json!(result?);
        cleanup?;
        data["ceiling"] = json!(ceiling);
        artifacts::atomic_json(&directory.join("maximum-beam.json"), &data)?;
        Ok(data)
    }
    pub fn downstream(
        &self,
        beam: u64,
        inference: &Value,
        env: &Environment,
        maximum: bool,
    ) -> Result<Value> {
        let mut options = self.request.inference.options.clone();
        options.calibration_full_frontier |= maximum;
        options.calibration_pipeline_seconds = options
            .calibration_pipeline_seconds
            .min(self.remaining()?.as_secs_f64());
        let execution = &self.request.execution;
        downstream_job::run(
            &downstream_job::Request {
                options,
                graph: self.request.inference.graph.clone(),
                runner: execution.runner.clone(),
                artifacts: self.request.inference.artifacts.clone(),
                inference: inference.clone(),
                environment: env.clone(),
                beam_width: beam,
                row_factor: execution.row_factor,
                hardware: execution.hardware.clone(),
                run_dir: self.directory("downstream")?,
                cwd: execution.cwd.clone(),
                physical_bytes: execution.physical_bytes,
                puzzle_id: execution.puzzle_id,
                grow: execution.grow,
                fixtures: execution.fixtures.clone(),
            },
            self.cancel.clone(),
        )
    }
    pub fn reference(
        &self,
        width: u64,
        pipeline: &Value,
        inference: &Value,
        env: &Environment,
    ) -> Result<Value> {
        self.verify()?;
        let budget =
            Duration::from_secs_f64(self.request.inference.options.calibration_pipeline_seconds)
                .min(self.remaining()?);
        let predicted = inference["estimate"]["low"]
            .as_f64()
            .unwrap_or(f64::INFINITY)
            * width as f64
            * 6.;
        if !predicted.is_finite() || predicted >= budget.as_secs_f64() {
            return Err(Error::Message(
                "working reference exceeds full-step verification budget".into(),
            ));
        }
        let directory = self.directory(&format!("working-reference-{width}"))?;
        let execution = &self.request.execution;
        let mut environment = env.clone();
        for key in [
            "BEAM_SOLVED_NEIGHBORHOOD_RADIUS",
            "BEAM_STREAM2_SUFFIX_RADIUS",
            "BEAM_SOLVE_BUCKET_MODE",
        ] {
            environment.insert(key.into(), "0".into());
        }
        let mut owner = PipelineProbe::launch_cancellable(
            Config {
                runner: execution.runner.path.clone(),
                environment,
                beam: width,
                world: self.world,
                cwd: execution.cwd.clone(),
                directory: directory.join("probe"),
                fixtures: vec![],
                physical_bytes: execution.physical_bytes,
                repeats: 6,
                puzzle_id: execution.puzzle_id,
                grow: execution.grow,
                gpu_uuids: serde_json::from_value(inference["signature"]["gpu_uuids"].clone())?,
                device_indices: serde_json::from_value(
                    inference["signature"]["device_indices"].clone(),
                )?,
            },
            budget,
            self.cancel.clone(),
        )?;
        let profile = Profile {
            name: "baseline".into(),
            environment: serde_json::from_value(pipeline["environment"].clone())?,
        };
        let result = owner.tune(&[profile], inference["parent_batch"].as_u64().unwrap());
        let cleanup = owner.close();
        let selected = result?;
        cleanup?;
        self.verify()?;
        let data = json!({"phase":"pipeline_measured","pipeline_verified":true,"measurement_scope":"full_requested_frontier",
            "requested_beam_width":width,"requested_beam_effective":width,"selection":selected.estimate.profile,
            "environment":selected.environment,"estimate":selected.estimate,"measurements":selected.measurements,
            "cache_hit":false,"policy":"fresh same-profile working-width comparison"});
        artifacts::atomic_json(&directory.join("selection.json"), &data)?;
        Ok(data)
    }
}
