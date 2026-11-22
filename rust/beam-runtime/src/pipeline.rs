//! Coarse native pipeline calibration. Complete steps are the timing unit.
use crate::{
    calibration::{tune_profiles, Environment, Profile, ProfileEvaluator, Selection},
    planning::PlanningSession,
    probes::ProbeCohort,
    telemetry::Telemetry,
};
use crate::{Error, Result};
use beam_core::{
    calibration::Measurement,
    planning::{rank_frontier_counts, RankPlan},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::{
    fs,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub runner: PathBuf,
    pub environment: Environment,
    pub beam: u64,
    pub world: u32,
    pub cwd: PathBuf,
    pub directory: PathBuf,
    pub fixtures: Vec<PathBuf>,
    pub physical_bytes: u64,
    pub repeats: usize,
    pub puzzle_id: u64,
    pub grow: bool,
    pub gpu_uuids: Vec<String>,
    pub device_indices: Vec<u32>,
}
pub struct PipelineProbe {
    config: Config,
    planner: Option<PlanningSession>,
    cancel: Arc<AtomicBool>,
    deadline: Instant,
    counter: u64,
    fastest: Option<Duration>,
    closed: bool,
}
impl PipelineProbe {
    pub fn launch(config: Config, timeout: Duration) -> Result<Self> {
        Self::launch_cancellable(config, timeout, Arc::new(AtomicBool::new(false)))
    }
    pub fn launch_cancellable(
        mut config: Config,
        timeout: Duration,
        cancel: Arc<AtomicBool>,
    ) -> Result<Self> {
        if config.beam == 0
            || !(1..=65535).contains(&config.world)
            || !(6..=16).contains(&config.repeats)
            || config.physical_bytes == 0
            || config.device_indices.len() != config.world as usize
            || (!config.grow && config.fixtures.len() != config.world as usize)
            || timeout.is_zero()
        {
            return Err(Error::Message(
                "invalid pipeline calibration geometry/timeout".into(),
            ));
        }
        config.runner = config.runner.canonicalize()?;
        config.cwd = config.cwd.canonicalize()?;
        if !config.directory.is_absolute() {
            config.directory = std::env::current_dir()?.join(&config.directory);
        }
        if !config.grow {
            for path in &mut config.fixtures {
                *path = path.canonicalize()?;
            }
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| Error::Message("pipeline deadline overflow".into()))?;
        fs::create_dir(&config.directory)?;
        let command = vec![
            config.runner.to_string_lossy().into_owned(),
            config.puzzle_id.to_string(),
            "1".into(),
            "1024".into(),
        ];
        let planner = PlanningSession::launch_cancellable(
            &command,
            &config.environment,
            config.world,
            &config.cwd,
            &config.directory.join("planning"),
            timeout,
            cancel.clone(),
        )?;
        Ok(Self {
            config,
            planner: Some(planner),
            cancel,
            deadline,
            counter: 0,
            fastest: None,
            closed: false,
        })
    }
    pub fn cancellation(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }
    fn remaining(&self) -> Result<Duration> {
        if self.closed || self.cancel.load(Ordering::Acquire) {
            return Err(Error::Message(
                "pipeline calibration cancelled or closed".into(),
            ));
        }
        self.deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or_else(|| Error::Message("pipeline calibration timed out".into()))
    }
    pub fn tune(&mut self, profiles: &[Profile], micro: u64) -> Result<Selection> {
        self.remaining()?;
        tune_profiles(self, profiles, self.config.beam, micro, self.deadline)
    }
    pub fn close(&mut self) -> Result<()> {
        self.closed = true;
        self.finish_planning()
    }
    fn run(&mut self, environment: &Environment) -> Result<(Vec<String>, PathBuf, Duration, bool)> {
        let remaining = self.remaining()?;
        let allowance = self
            .fastest
            .map(|d| d.saturating_mul(4).max(Duration::from_secs(30)))
            .unwrap_or(remaining)
            .min(remaining);
        self.counter = self
            .counter
            .checked_add(1)
            .ok_or_else(|| Error::Message("pipeline trial counter overflow".into()))?;
        let directory = self.config.directory.join(self.counter.to_string());
        fs::create_dir(&directory)?;
        fs::create_dir(directory.join("test_results"))?;
        let mut env = self.config.environment.clone();
        env.extend(environment.clone());
        for key in [
            "BEAM_BENCHMARK_PLAN_ONLY",
            "BEAM_BENCHMARK_FRONTIER_FILE",
            "BEAM_BENCHMARK_GROW_STEPS",
            "BEAM_CALIBRATION_COMPONENT_ONLY",
            "BEAM_CALIBRATION_PLAN_SESSION",
            "BEAM_SESSION_DIR",
        ] {
            env.remove(key);
        }
        for (key, value) in [
            (
                "BEAM_NCCL_RUN_ID",
                format!("pipeline-{}-{}", std::process::id(), directory.display()),
            ),
            (
                "BEAM_NCCL_ID_FILE",
                directory.join("nccl-id.bin").to_string_lossy().into_owned(),
            ),
            (
                "BEAM_BENCHMARK_FRONTIER_REPEATS",
                self.config.repeats.to_string(),
            ),
            (
                "BEAM_HISTORY_DIR",
                directory.join("history").to_string_lossy().into_owned(),
            ),
            (
                "BEAM_HISTORY_DISK_PATH",
                directory.join("history").to_string_lossy().into_owned(),
            ),
        ] {
            env.insert(key.into(), value);
        }
        if self.config.grow {
            env.insert(
                "BEAM_BENCHMARK_GROW_STEPS".into(),
                self.config.repeats.to_string(),
            );
        }
        let mut commands = Vec::new();
        let mut environments = Vec::new();
        let depth = if self.config.grow {
            32 + self.config.repeats
        } else {
            self.config.repeats
        };
        for rank in 0..self.config.world {
            let mut rank_env = env.clone();
            for (key, value) in [
                ("RANK", rank),
                ("LOCAL_RANK", rank),
                ("WORLD_SIZE", self.config.world),
                ("LOCAL_WORLD_SIZE", self.config.world),
            ] {
                rank_env.insert(key.into(), value.to_string());
            }
            if !self.config.grow {
                rank_env.insert(
                    "BEAM_BENCHMARK_FRONTIER_FILE".into(),
                    self.config.fixtures[rank as usize]
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            commands.push(vec![
                self.config.runner.to_string_lossy().into_owned(),
                self.config.puzzle_id.to_string(),
                depth.to_string(),
                self.config.beam.to_string(),
                self.config.world.to_string(),
                rank.to_string(),
            ]);
            environments.push(rank_env);
        }
        let began = Instant::now();
        let telemetry = Telemetry::start()?;
        let run = (|| {
            let mut group = ProbeCohort::launch(
                &commands,
                &environments,
                &directory,
                &directory.join("ranks"),
                allowance,
            )?;
            group.bind_cancellation(self.cancel.clone());
            group.finish(allowance.min(self.remaining()?))?;
            (0..self.config.world)
                .map(|rank| {
                    let text =
                        fs::read_to_string(directory.join(format!("ranks/rank-{rank}.log")))?;
                    if text.contains("production_runner_error=") {
                        return Err(Error::Message(format!(
                            "native calibration failed in rank {rank}"
                        )));
                    }
                    Ok(text)
                })
                .collect::<Result<Vec<_>>>()
        })();
        let receipt = telemetry.finish()?;
        fs::write(
            directory.join("gpu-telemetry.json"),
            serde_json::to_vec_pretty(&receipt)?,
        )?;
        let verified = receipt.verified(&self.config.gpu_uuids, &self.config.device_indices);
        let texts = run.map_err(|error| {
            crate::probes::rank_error(error, &directory.join("ranks"), self.config.world as usize)
        })?;
        Ok((texts, directory, began.elapsed(), verified))
    }
}
impl ProfileEvaluator for PipelineProbe {
    fn admit(&mut self, environment: &Environment) -> Result<Vec<Value>> {
        self.remaining()?;
        self.planner
            .as_mut()
            .ok_or_else(|| Error::Message("pipeline planning owner was released".into()))?
            .admit(self.config.beam, environment)
    }
    fn finish_planning(&mut self) -> Result<()> {
        if let Some(mut planner) = self.planner.take() {
            planner.close()?;
        }
        Ok(())
    }
    fn measure(&mut self, environment: &Environment, plans: &[Value]) -> Result<Vec<Measurement>> {
        if self.planner.is_some() {
            return Err(Error::Message(
                "release planning owner before full-step measurement".into(),
            ));
        }
        let typed: Vec<RankPlan> = plans
            .iter()
            .cloned()
            .map(serde_json::from_value)
            .collect::<std::result::Result<_, _>>()?;
        let counts = rank_frontier_counts(&typed, self.config.beam)?;
        if counts.len() != self.config.world as usize {
            return Err(Error::Message("pipeline rank cohort changed".into()));
        }
        if !self.config.grow {
            for (path, count) in self.config.fixtures.iter().zip(&counts) {
                let required = count
                    .checked_mul(self.config.physical_bytes)
                    .ok_or_else(|| Error::Message("frontier fixture size overflow".into()))?;
                if fs::metadata(path)?.len() != required {
                    return Err(Error::Message(format!(
                        "frontier fixture must fill exact rank: required={required} actual={}",
                        fs::metadata(path)?.len()
                    )));
                }
            }
        }
        let (texts, directory, wall, telemetry_verified) = self.run(environment)?;
        let mut seconds = Vec::new();
        for (rank, (text, count)) in texts.iter().zip(&counts).enumerate() {
            let fields = parse_plan(text)?;
            if fields.get("GLOBAL_BEAM_WIDTH_EFFECTIVE") != Some(&self.config.beam) {
                return Err(Error::Message(
                    "full-step runner changed exact frontier".into(),
                ));
            }
            if self.config.grow
                && fields.get("calibration_saturated_steps") != Some(&(self.config.repeats as u64))
            {
                return Err(Error::Message(
                    "natural search did not complete saturated frontier cohort".into(),
                ));
            }
            for line in text.lines().filter(|l| l.starts_with("calibration_depth=")) {
                let row: BTreeMap<_, _> = line
                    .split_whitespace()
                    .filter_map(|v| v.split_once('='))
                    .collect();
                if row.get("rank").and_then(|n| n.parse::<usize>().ok()) != Some(rank)
                    || row
                        .get("next_frontier_size")
                        .and_then(|n| n.parse::<u64>().ok())
                        != Some(*count)
                {
                    return Err(Error::Message(
                        "full-step output frontier/rank mismatch".into(),
                    ));
                }
            }
            seconds.push(parse_depths(text, *count, self.config.repeats)?);
        }
        self.fastest = Some(self.fastest.map_or(wall, |old| old.min(wall)));
        let rows: Vec<_> = (1..self.config.repeats)
            .map(|i| Measurement {
                profile: String::new(),
                parents: self.config.beam,
                seconds_by_rank: seconds.iter().map(|r| r[i]).collect(),
                correctness_passed: true,
                memory_admitted: true,
                throttled: !telemetry_verified,
            })
            .collect();
        fs::write(
            directory.join("measurements.json"),
            serde_json::to_vec_pretty(&rows)?,
        )?;
        Ok(rows)
    }
}
impl Drop for PipelineProbe {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub fn parse_plan(text: &str) -> Result<BTreeMap<String, u64>> {
    let mut values = BTreeMap::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.is_empty()
            || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
            || !key.as_bytes()[0].is_ascii_alphabetic() && key.as_bytes()[0] != b'_'
        {
            continue;
        }
        if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let number = value
            .parse::<u64>()
            .map_err(|_| Error::Message(format!("native plan integer overflow: {key}")))?;
        if values
            .insert(key.into(), number)
            .is_some_and(|old| old != number)
        {
            return Err(Error::Message(format!("native plan changes {key}")));
        }
    }
    if !values.contains_key("GLOBAL_BEAM_WIDTH_EFFECTIVE") {
        return Err(Error::Message("native plan missing effective beam".into()));
    }
    Ok(values)
}
pub fn parse_depths(text: &str, expected_parents: u64, repeats: usize) -> Result<Vec<f64>> {
    let mut rows = Vec::new();
    for line in text.lines().filter(|l| l.starts_with("calibration_depth=")) {
        let fields: BTreeMap<_, _> = line
            .split_whitespace()
            .filter_map(|f| f.split_once('='))
            .collect();
        let number = |key: &str| fields.get(key).and_then(|v| v.parse::<u64>().ok());
        if number("calibration_depth") != Some(rows.len() as u64) {
            return Err(Error::Message(
                "missing, duplicate or reordered calibration depth".into(),
            ));
        }
        if number("parents") != Some(expected_parents) {
            return Err(Error::Message(
                "frontier workload changed during calibration".into(),
            ));
        }
        let seconds = fields
            .get("depth_sec")
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|s| s.is_finite() && *s > 0.)
            .ok_or_else(|| Error::Message("invalid complete-depth timing".into()))?;
        rows.push(seconds);
    }
    if rows.len() != repeats {
        return Err(Error::Message(
            "native run did not complete all calibration depths".into(),
        ));
    }
    Ok(rows)
}
