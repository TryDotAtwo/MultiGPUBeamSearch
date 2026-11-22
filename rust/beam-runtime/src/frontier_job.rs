//! One coarse job owns inference admission, downstream policy and symbolic width.
use crate::{
    artifacts::{self, Artifact},
    frontier_native::Owner,
    inference_admission, inference_job, profile_cache, working_capacity, Error, Result,
};
use beam_core::{
    capacity::{frontier_ceiling, BeamWidth, FrontierCeiling},
    profile_policy::Hardware,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::{atomic::AtomicBool, Arc},
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Execution {
    pub runner: Artifact,
    pub row_factor: u64,
    pub hardware: Hardware,
    pub physical_bytes: u64,
    pub puzzle_id: u64,
    pub grow: bool,
    pub fixtures: Vec<Artifact>,
    pub cwd: PathBuf,
    pub gpu_memory_bytes: Vec<u64>,
    /// Optional bound for an explicitly bounded diagnostic/coarse request.
    /// The public adapter leaves it unset; it is not the fixture state budget.
    #[serde(default)]
    pub frontier_ceiling: Option<u64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub beam_width: BeamWidth,
    pub inference: inference_job::Request,
    pub execution: Execution,
}
impl Request {
    pub fn capacity_ceiling(&self) -> Result<FrontierCeiling> {
        let world = self.inference.signature["world_size"]
            .as_u64()
            .filter(|n| *n > 0 && *n <= 65535)
            .ok_or_else(|| Error::Message("invalid capacity cohort".into()))?
            as u32;
        Ok(frontier_ceiling(
            &self.inference.graph,
            world,
            self.execution.physical_bytes,
            &self.execution.gpu_memory_bytes,
            self.execution.frontier_ceiling,
        )?)
    }
}
pub fn validate(request: &Request) -> Result<()> {
    let prepared = inference_job::prepare(&request.inference)?;
    let execution = &request.execution;
    if !request.inference.options.autotune
        || execution.hardware.sm_count == 0
        || (execution.row_factor != 1
            && execution.row_factor != request.inference.graph.generators.len() as u64)
        || execution.physical_bytes < request.inference.graph.center.len() as u64
        || execution.physical_bytes % 16 != 0
        || execution.physical_bytes > 4096
        || execution.gpu_memory_bytes.len() != prepared.world as usize
        || execution.gpu_memory_bytes.contains(&0)
        || execution.frontier_ceiling == Some(0)
        || (request.beam_width == BeamWidth::Maximum && !execution.grow)
        || (matches!(request.beam_width, BeamWidth::Exact(0)))
    {
        return Err(Error::Message("invalid frontier calibration contract; maximum requires native natural-growth verification".into()));
    }
    Ok(())
}
pub fn refine_maximum(
    mut capacity: Value,
    mut previous: Value,
    max_batch: u64,
    mut tune: impl FnMut(u64, usize) -> Result<Value>,
    mut discover: impl FnMut(&Value, usize) -> Result<Value>,
) -> Result<(Value, Value)> {
    let mut history = vec![];
    for round in 0..3 {
        let beam = capacity["effective_beam"]
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or_else(|| Error::Message("missing admitted maximum width".into()))?;
        let measured = tune(beam, round)?;
        if measured["signature"]["requested_beam_width"].as_u64() != Some(beam) {
            return Err(Error::Message(
                "maximum inference is not bound to exact frontier".into(),
            ));
        }
        let calibration = profile_cache::normalize(&measured, &measured["signature"], max_batch)?;
        let old = previous["parent_batch"].to_string();
        if calibration["rejected"][&old]
            .as_str()
            .is_some_and(|s| s.contains("GPU telemetry"))
        {
            return Err(Error::Message("cannot expand maximum by dropping a previous winner with missing/throttled GPU telemetry".into()));
        }
        let changed = ["parent_batch", "reserve_bytes"]
            .iter()
            .any(|key| calibration[*key] != previous[*key]);
        if changed {
            capacity = discover(&calibration, round)?;
        }
        history.push(
            json!({"beam":beam,"parent_batch":calibration["parent_batch"],
            "reserve_bytes":calibration["reserve_bytes"],"readmitted":changed}),
        );
        if capacity["effective_beam"].as_u64() == Some(beam) {
            capacity["inference_refinement"] = json!(history);
            return Ok((capacity, calibration));
        }
        previous = calibration;
    }
    Err(Error::Message(
        "maximum frontier/inference calibration did not converge".into(),
    ))
}
pub fn run(request: &Request, cancel: Arc<AtomicBool>) -> Result<Value> {
    validate(request)?;
    let started = Instant::now();
    let owner = Owner::new(request, cancel)?;
    let result: Result<Value> = (|| {
        owner.verify()?;
        let maximum = request.beam_width == BeamWidth::Maximum;
        let initial_width = match request.beam_width {
            BeamWidth::Exact(n) => n,
            BeamWidth::Maximum => 1_048_576,
        };
        let initial = owner.inference(initial_width, maximum, "initial")?;
        let (beam, inference, mut capacity) = if maximum {
            let seed = inference_admission::capacity_seed(
                &initial,
                request.inference.options.calibration_max_batch,
            )?;
            let capacity = owner.discover(&seed, true, "initial-capacity")?;
            let (capacity, calibration) = refine_maximum(
                capacity,
                seed,
                request.inference.options.calibration_max_batch,
                |width, index| owner.inference(width, false, &format!("maximum-inference-{index}")),
                |chosen, index| owner.discover(chosen, true, &format!("maximum-capacity-{index}")),
            )?;
            (
                capacity["effective_beam"].as_u64().unwrap(),
                calibration,
                capacity,
            )
        } else {
            let capacity = if request.inference.options.calibration_discover_capacity {
                let seed = inference_admission::capacity_seed(
                    &initial,
                    request.inference.options.calibration_max_batch,
                )?;
                let mut capacity = owner.discover(&seed, false, "diagnostic-capacity")?;
                capacity["requested_execution_beam"] = json!(initial_width);
                capacity["execution_width_changed"] = json!(false);
                capacity
            } else {
                Value::Null
            };
            (
                initial_width,
                owner.admit_fixed(initial_width, &initial)?,
                capacity,
            )
        };
        artifacts::atomic_json(&owner.root.join("capacity.json"), &capacity)?;
        let mut environment = owner.environment(&inference)?;
        if maximum {
            environment.extend(serde_json::from_value::<crate::calibration::Environment>(
                capacity["profile"].clone(),
            )?);
        }
        let mut pipeline = owner.downstream(beam, &inference, &environment, maximum)?;
        let mut width = beam;
        let mut working = Value::Null;
        if maximum {
            working_capacity::require_full_step(beam, owner.world, &pipeline)?;
            capacity["full_step_boundary_beam"] = json!(beam);
            capacity["full_step_verified"] = json!(true);
            capacity["capacity_scope"] = json!("full-step verified; working efficiency pending");
            artifacts::atomic_json(&owner.root.join("capacity.json"), &capacity)?;
            let selection = working_capacity::choose(
                beam,
                pipeline,
                inference["parent_batch"].as_u64().unwrap(),
                owner.world,
                |width, profile| owner.reference(width, profile, &inference, &environment),
            )?;
            width = selection.beam;
            working = json!(selection);
            pipeline = selection.pipeline;
            // A smaller measured working width does not relabel the inference
            // calibration's original width or the admitted memory ceiling.
            capacity["effective_beam"] = json!(width);
            capacity["capacity_scope"] =
                json!("full-step verified; bounded measured working comparison");
            capacity["working_capacity"] = working.clone();
        }
        environment.extend(serde_json::from_value::<crate::calibration::Environment>(
            pipeline["environment"].clone(),
        )?);
        owner.verify()?;
        let data = json!({"policy":"rust-frontier-owner-v1","requested_beam_width":request.beam_width,
            "effective_beam_width":width,"inference":inference,"pipeline":pipeline,"environment":environment,
            "capacity":capacity,"working_capacity":working,"wall_seconds":started.elapsed().as_secs_f64()});
        artifacts::atomic_json(&owner.root.join("selection.json"), &data)?;
        Ok(data)
    })();
    if let Err(error) = &result {
        artifacts::atomic_json(
            &owner.root.join("failure.json"),
            &json!({"error":error.to_string()}),
        )?;
    }
    result
}
