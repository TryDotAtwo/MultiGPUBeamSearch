//! Persistent exact-capacity planner; no rank enters NCCL before all are ready.
use crate::{probes::ProbeCohort, Error, Result};
use beam_core::planning::{validate_rank_plans, RankPlan, TUNABLE_KEYS};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

pub struct PlanningSession {
    group: ProbeCohort,
    inference_micro: Option<String>,
    components: bool,
    world: usize,
    deadline: Instant,
}
impl PlanningSession {
    /// `command` contains runner, puzzle, depth and initial beam. World and rank
    /// are appended; every process owns its own CUDA pointers and NCCL context.
    pub fn launch(
        command: &[String],
        environment: &BTreeMap<String, String>,
        world: u32,
        cwd: &Path,
        directory: &Path,
        timeout: Duration,
    ) -> Result<Self> {
        if world == 0 {
            return Err(Error::Message("empty planning cohort".into()));
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| Error::Message("planning deadline overflow".into()))?;
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let id = format!(
            "plan-session-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        );
        let directory = if directory.is_absolute() {
            directory.to_path_buf()
        } else {
            std::env::current_dir()?.join(directory)
        };
        let mut commands = Vec::new();
        let mut environments = Vec::new();
        for rank in 0..world {
            let mut args = command.to_vec();
            args.extend([world.to_string(), rank.to_string()]);
            commands.push(args);
            let mut env = environment.clone();
            for (key, value) in [
                ("BEAM_CALIBRATION_PLAN_SESSION", "1".into()),
                ("BEAM_BENCHMARK_PLAN_ONLY", "1".into()),
                ("BEAM_NCCL_RUN_ID", id.clone()),
                (
                    "BEAM_NCCL_ID_FILE",
                    directory.join("nccl-id.bin").to_string_lossy().into_owned(),
                ),
                ("RANK", rank.to_string()),
                ("LOCAL_RANK", rank.to_string()),
                ("WORLD_SIZE", world.to_string()),
                ("LOCAL_WORLD_SIZE", world.to_string()),
            ] {
                env.insert(key.into(), value);
            }
            environments.push(env);
        }
        let mut group = ProbeCohort::launch(&commands, &environments, cwd, &directory, timeout)?;
        let ready = match group.collect(timeout) {
            Ok(ready) => ready,
            Err(error) => {
                let _ = group.close();
                let failures: Vec<_> = (0..world)
                    .filter_map(|rank| {
                        std::fs::read_to_string(directory.join(format!("rank-{rank}.log"))).ok()
                    })
                    .flat_map(|text| {
                        text.lines()
                            .filter_map(|line| {
                                line.strip_prefix("production_runner_error=")
                                    .map(str::to_owned)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                if !failures.is_empty()
                    && failures
                        .iter()
                        .all(|s| s == "insufficient blend GPU reserve")
                {
                    return Err(Error::CapacityRejected(
                        "native planner initialization rejected measured blend reserve".into(),
                    ));
                }
                return Err(error);
            }
        };
        let mut capabilities = Vec::new();
        for (rank, row) in ready.iter().enumerate() {
            if row["ready"] != true || row["rank"].as_u64() != Some(rank as u64) {
                return Err(Error::Message(
                    "native planning service did not acknowledge its rank".into(),
                ));
            }
            capabilities.push(row["component_protocol"] == "persistent-exact-capacity-v1");
        }
        let components = capabilities.iter().all(|v| *v);
        if !components && capabilities.iter().any(|v| *v) {
            return Err(Error::Message(
                "mixed native component service cohort".into(),
            ));
        }
        Ok(Self {
            group,
            inference_micro: environment.get("BEAM_ENSEMBLE_INFERENCE_MICRO").cloned(),
            components,
            world: world as usize,
            deadline,
        })
    }
    pub fn supports_components(&self) -> bool {
        self.components
    }
    pub fn pids(&self) -> Vec<u32> {
        self.group.pids()
    }
    pub fn cancellation(&self) -> Arc<AtomicBool> {
        self.group.cancellation()
    }
    pub fn exit_codes(&mut self) -> Result<Vec<Option<i32>>> {
        self.group.exit_codes()
    }
    pub fn set_deadline(&mut self, timeout: Duration) -> Result<()> {
        self.deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| Error::Message("planning deadline overflow".into()))?;
        self.group.set_deadline(timeout)
    }
    fn overrides(
        &self,
        environment: &BTreeMap<String, String>,
    ) -> Result<BTreeMap<String, String>> {
        let mut env = environment.clone();
        if let Some(value) = env.remove("BEAM_ENSEMBLE_INFERENCE_MICRO") {
            if Some(&value) != self.inference_micro.as_ref() {
                return Err(Error::Message(
                    "planner cannot change frozen model inference microbatch".into(),
                ));
            }
        }
        for (key, value) in &env {
            if !TUNABLE_KEYS.contains(&key.as_str()) || !value.parse::<u64>().is_ok_and(|n| n > 0) {
                return Err(Error::Message(format!(
                    "invalid planning override {key}; cannot change semantic/capacity policy"
                )));
            }
        }
        Ok(env)
    }
    fn exchange(&mut self, request: Value) -> Result<Vec<Value>> {
        self.group.send(&vec![request; self.world])?;
        self.group
            .collect(self.deadline.saturating_duration_since(Instant::now()))
    }
    pub fn admit(
        &mut self,
        beam: u64,
        environment: &BTreeMap<String, String>,
    ) -> Result<Vec<Value>> {
        if beam == 0 {
            return Err(Error::Message("beam must be positive".into()));
        }
        let env = self.overrides(environment)?;
        let rows = self.exchange(json!({"beam":beam,"environment":env}))?;
        let reasons: Vec<_> = rows
            .iter()
            .filter(|r| r["capacity_rejected"] == true)
            .map(|r| r["reason"].as_str().unwrap_or("native capacity rejected"))
            .collect();
        if !reasons.is_empty() {
            return Err(Error::CapacityRejected(reasons.join("; ")));
        }
        if rows.iter().any(|r| r["admitted"] != true) {
            return Err(Error::Message(
                "native planner returned invalid admission receipt".into(),
            ));
        }
        let values: Vec<_> = rows.iter().map(|r| r["plan"].clone()).collect();
        let plans: Vec<RankPlan> = values
            .iter()
            .cloned()
            .map(serde_json::from_value)
            .collect::<std::result::Result<_, _>>()?;
        validate_rank_plans(&plans, beam)?;
        for (rank, plan) in plans.iter().enumerate() {
            if plan.rank != Some(rank as u32) {
                return Err(Error::Message(
                    "planning response has wrong communicator rank".into(),
                ));
            }
        }
        Ok(values)
    }
    pub fn measure_components(
        &mut self,
        beam: u64,
        environment: &BTreeMap<String, String>,
    ) -> Result<Vec<Value>> {
        if !self.components {
            return Err(Error::Message(
                "native component session unsupported".into(),
            ));
        }
        if beam == 0 {
            return Err(Error::Message("beam must be positive".into()));
        }
        let env = self.overrides(environment)?;
        let ready =
            self.exchange(json!({"beam":beam,"environment":env,"prepare_component":true}))?;
        if ready
            .iter()
            .enumerate()
            .any(|(rank, r)| r["prepared"] != true || r["rank"].as_u64() != Some(rank as u64))
        {
            let cancelled = self.exchange(json!({"cancel_component":true}))?;
            if cancelled
                .iter()
                .enumerate()
                .any(|(rank, r)| r["cancelled"] != true || r["rank"].as_u64() != Some(rank as u64))
            {
                let _ = self.group.close();
                return Err(Error::Message(
                    "component cancellation was not acknowledged by every rank".into(),
                ));
            }
            return Err(Error::CapacityRejected(
                "component allocation was not prepared on every rank".into(),
            ));
        }
        self.exchange(json!({"run_component":true}))
    }
    pub fn close(&mut self) -> Result<()> {
        self.group.close()
    }
    pub fn find_capacity(
        &mut self,
        profiles: &[BTreeMap<String, String>],
        upper: u64,
        alignment: u64,
    ) -> Result<crate::calibration::CapacityResult> {
        if profiles
            .iter()
            .any(|p| TUNABLE_KEYS.iter().any(|key| !p.contains_key(*key)))
        {
            return Err(Error::Message(
                "capacity search requires all five scheduling fields frozen".into(),
            ));
        }
        crate::calibration::find_capacity(profiles, upper, alignment, self.deadline, |beam, env| {
            self.admit(beam, env)
        })
    }
}
