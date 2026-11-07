//! Whole downstream calibration job: identity, admission, cache and measurement.
use crate::{
    artifacts::{self, Artifact},
    calibration::Environment,
    component_probe::NativeComponents,
    component_tuning::{self, ComponentEvaluator, TuneConfig},
    pipeline_cache, profile_cache, Error, Result,
};
use beam_core::{config::NativeOptions, profile_policy::Hardware, search::PermutationGraph};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fs,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
pub const POLICY: &str = "rust-downstream-owner-v1";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub options: NativeOptions,
    pub graph: PermutationGraph,
    pub runner: Artifact,
    pub artifacts: Vec<Artifact>,
    pub inference: Value,
    pub environment: Environment,
    pub beam_width: u64,
    pub row_factor: u64,
    pub hardware: Hardware,
    pub run_dir: PathBuf,
    pub cwd: PathBuf,
    pub physical_bytes: u64,
    pub puzzle_id: u64,
    pub grow: bool,
    pub fixtures: Vec<Artifact>,
}
pub struct Prepared {
    pub identity: Value,
    pub key: String,
    pub tuning: TuneConfig,
    pub environment: Environment,
    pub overrides: Environment,
}
pub fn prepare(request: &Request) -> Result<Prepared> {
    request.options.validate()?;
    request.graph.validate()?;
    let signature = &request.inference["signature"];
    let world = signature["world_size"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= 65535)
        .ok_or_else(|| Error::Message("invalid downstream world size".into()))?;
    let devices: Vec<u32> = serde_json::from_value(signature["device_indices"].clone())?;
    let uuids: Vec<String> = serde_json::from_value(signature["gpu_uuids"].clone())?;
    let moves = request.graph.generators.len() as u32;
    if request.beam_width == 0
        || devices.len() != world as usize
        || devices.iter().collect::<BTreeSet<_>>().len() != devices.len()
        || request.artifacts.is_empty()
        || json!(request.artifacts) != signature["artifact_files"]
        || signature["requested_beam_width"].as_u64() != Some(request.beam_width)
        || signature["graph_contract_sha256"] != profile_cache::cache_key(&json!(request.graph))?
        || (request.row_factor != 1 && request.row_factor != moves as u64)
        || request.hardware.sm_count == 0
        || request.physical_bytes < request.graph.center.len() as u64
        || request.physical_bytes % 16 != 0
        || request.physical_bytes > 4096
        || (request.options.calibration_full_frontier
            && !request.grow
            && request.fixtures.len() != world as usize)
    {
        return Err(Error::Message(
            "inconsistent downstream graph/model/geometry".into(),
        ));
    }
    let inference = profile_cache::normalize(
        &request.inference,
        signature,
        request.options.calibration_max_batch,
    )?;
    let micro = inference["parent_batch"]
        .as_u64()
        .ok_or_else(|| Error::Message("missing inference batch".into()))?;
    let seconds = inference["estimate"]["median"]
        .as_f64()
        .ok_or_else(|| Error::Message("missing verified inference timing".into()))?;
    let outer = micro
        .checked_mul(request.row_factor)
        .ok_or_else(|| Error::Message("outer row batch overflow".into()))?;
    let overrides: Environment = [
        ("BEAM_B_MICRO".into(), outer.to_string()),
        ("BEAM_ENSEMBLE_INFERENCE_MICRO".into(), micro.to_string()),
    ]
    .into();
    let mut environment = request.environment.clone();
    environment.extend(overrides.clone());
    // Calibration has no solved-neighborhood shortcuts. Ordinary search keeps
    // its own original environment and exact requested semantics.
    for key in [
        "BEAM_SOLVED_NEIGHBORHOOD_RADIUS",
        "BEAM_STREAM2_SUFFIX_RADIUS",
        "BEAM_SOLVE_BUCKET_MODE",
    ] {
        environment.insert(key.into(), "0".into());
    }
    environment.insert(
        "BEAM_ENSEMBLE_RESERVE_BYTES".into(),
        inference["reserve_bytes"].to_string(),
    );
    let policy_env: Environment = environment
        .iter()
        .filter(|(k, _)| {
            (k.starts_with("BEAM_")
                || k.starts_with("NCCL_")
                || k.as_str() == "CUDA_VISIBLE_DEVICES")
                && ![
                    "BEAM_NCCL_RUN_ID",
                    "BEAM_PUZZLE_INFO_JSON",
                    "BEAM_TEST_CSV",
                    "BEAM_GENERATOR_PATH",
                ]
                .contains(&k.as_str())
                && !["_PATH", "_FILE", "_DIR"]
                    .iter()
                    .any(|suffix| k.ends_with(suffix))
        })
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let tuning = TuneConfig {
        beam: request.beam_width,
        moves,
        micro,
        row_factor: request.row_factor,
        inference_seconds: seconds,
        hardware: request.hardware.clone(),
        gpu_uuids: uuids,
        device_indices: devices,
    };
    let identity = json!({"schema":1,"policy":POLICY,"proposal_policy":beam_core::profile_policy::POLICY,
        "runner":request.runner,"graph":request.graph,"artifacts":request.artifacts,"signature":signature,
        "tuning":tuning,"reserve_bytes":inference["reserve_bytes"],"policy_environment":policy_env,
        "full_frontier":request.options.calibration_full_frontier,"physical_bytes":request.physical_bytes,
        "puzzle_id":request.puzzle_id,"grow":request.grow,"fixtures":request.fixtures});
    let key = profile_cache::cache_key(&identity)?;
    Ok(Prepared {
        identity,
        key,
        tuning,
        environment,
        overrides,
    })
}
fn remaining(deadline: Instant, cancel: &AtomicBool) -> Result<Duration> {
    if cancel.load(Ordering::Acquire) {
        return Err(Error::Message("downstream calibration cancelled".into()));
    }
    deadline
        .checked_duration_since(Instant::now())
        .filter(|d| !d.is_zero())
        .ok_or_else(|| Error::Message("downstream calibration timed out".into()))
}
pub fn run(request: &Request, cancel: Arc<AtomicBool>) -> Result<Value> {
    let started = Instant::now();
    let budget = Duration::try_from_secs_f64(request.options.calibration_pipeline_seconds)
        .map_err(|_| Error::Message("invalid downstream calibration budget".into()))?;
    let deadline = started
        .checked_add(budget)
        .ok_or_else(|| Error::Message("downstream deadline overflow".into()))?;
    remaining(deadline, &cancel)?;
    let prepared = prepare(request)?;
    request.runner.verify()?;
    artifacts::verify_all(&request.artifacts)?;
    artifacts::verify_all(&request.fixtures)?;
    remaining(deadline, &cancel)?;
    let directory = request.run_dir.join("pipeline-calibration");
    fs::create_dir(&directory)?;
    let command = vec![
        request.runner.path.to_string_lossy().into_owned(),
        request.puzzle_id.to_string(),
        "1".into(),
        "1024".into(),
    ];
    let mut owner = NativeComponents::launch(
        &command,
        &prepared.environment,
        prepared.tuning.device_indices.len() as u32,
        request.beam_width,
        &request.cwd,
        &directory.join("components"),
        remaining(deadline, &cancel)?,
        cancel.clone(),
    )?;
    let cache = request
        .options
        .cache_dir
        .join("pipeline-profiles")
        .join(format!("{}-components.json", prepared.key));
    let hit =
        pipeline_cache::read_components(&cache, &prepared.identity, &prepared.tuning, |env| {
            owner.admit(env)
        })?;
    let (components, current) = if let Some(hit) = hit {
        owner.close()?;
        (hit.selection, Some(hit.current_rank_plans))
    } else {
        let initial = owner.admit(&prepared.overrides)?;
        let baseline = component_tuning::normalized_environment(&prepared.tuning, &initial)?;
        (
            component_tuning::tune_components(&mut owner, &prepared.tuning, &baseline, deadline)?,
            None,
        )
    };
    remaining(deadline, &cancel)?;
    let mut data = if request.options.calibration_full_frontier {
        crate::downstream_full::run(
            request,
            &prepared,
            &components,
            &directory,
            deadline,
            cancel.clone(),
        )?
    } else {
        serde_json::to_value(&components)?
    };
    remaining(deadline, &cancel)?;
    request.runner.verify()?;
    artifacts::verify_all(&request.artifacts)?;
    artifacts::verify_all(&request.fixtures)?;
    remaining(deadline, &cancel)?;
    if components.search_complete && !components.cache_hit {
        pipeline_cache::write_components(
            &cache,
            &prepared.identity,
            &prepared.tuning,
            &components,
        )?;
    }
    data["requested_beam_width"] = json!(request.beam_width);
    data["cache_identity"] = prepared.identity;
    data["calibration_wall_seconds"] = json!(started.elapsed().as_secs_f64());
    if let Some(plans) = current {
        data["current_component_rank_plans"] = json!(plans);
    }
    artifacts::atomic_json(&directory.join("selection.json"), &data)?;
    Ok(data)
}
