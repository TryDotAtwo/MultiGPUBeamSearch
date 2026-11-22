//! Complete calibration request: bounds, input preparation, cache and execution.
use crate::{
    artifacts::{self, Artifact},
    calibration::Environment,
    inference::{InferencePool, PoolConfig, Protocol},
    inference_tuning::TuneConfig,
    profile_cache, telemetry, Error, Result,
};
use beam_core::{config::NativeOptions, search::PermutationGraph};
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
pub const POLICY: &str = "rust-calibration-owner-v1";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub options: NativeOptions,
    pub graph: PermutationGraph,
    pub helper: Artifact,
    pub artifacts: Vec<Artifact>,
    pub probe_manifest: Value,
    pub ensemble: bool,
    pub signature: Value,
    pub beam_width: u64,
    pub protocol: Protocol,
    pub capacity_bootstrap: bool,
    pub run_dir: PathBuf,
    pub environment: Environment,
    pub shared_profile: Option<Value>,
}
pub struct Prepared {
    pub signature: Value,
    pub key: String,
    pub samples: Vec<Vec<u8>>,
    pub capacity: u64,
    pub cap: u64,
    pub parents: u64,
    pub world: u32,
    pub devices: Vec<u32>,
    pub uuids: Vec<String>,
    pub shared_batch: Option<u64>,
}
fn graph_samples(graph: &PermutationGraph) -> Vec<Vec<u8>> {
    let mut state = graph.center.clone();
    let mut scratch = vec![0; state.len()];
    let mut seed = 20261009u64;
    let mut samples = Vec::with_capacity(128);
    for _ in 0..128 {
        for _ in 0..16 {
            seed = seed.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            let index = ((z as u128 * graph.generators.len() as u128) >> 64) as usize;
            for (value, source) in scratch.iter_mut().zip(&graph.generators[index]) {
                *value = state[*source as usize];
            }
            std::mem::swap(&mut state, &mut scratch);
        }
        samples.push(state.clone());
    }
    samples
}
pub fn prepare(request: &Request) -> Result<Prepared> {
    request.options.validate()?;
    request.graph.validate()?;
    if request.beam_width == 0
        || !request.signature.is_object()
        || !request.probe_manifest.is_object()
        || request.artifacts.is_empty()
    {
        return Err(Error::Message("invalid prepared inference request".into()));
    }
    let world = request.signature["world_size"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= 65535)
        .ok_or_else(|| Error::Message("invalid inference world size".into()))?
        as u32;
    let devices: Vec<u32> = serde_json::from_value(request.signature["device_indices"].clone())?;
    let uuids: Vec<String> = serde_json::from_value(request.signature["gpu_uuids"].clone())?;
    if devices.len() != world as usize
        || devices.iter().collect::<BTreeSet<_>>().len() != devices.len()
        || (!uuids.is_empty() && uuids.len() != devices.len())
    {
        return Err(Error::Message("invalid inference device identities".into()));
    }
    let maximum = if request.capacity_bootstrap {
        request.options.calibration_max_batch.min(32)
    } else {
        request.options.calibration_max_batch
    };
    let local = request.beam_width.div_ceil(world as u64);
    let capacity = local.min(request.options.calibration_max_batch);
    let cap = local.min(maximum);
    let parents = local.min(8192.max(capacity));
    let samples = graph_samples(&request.graph);
    let mut signature = request.signature.clone();
    let fields = signature.as_object_mut().unwrap();
    for (key, value) in [
        ("schema", json!(10)),
        ("search_policy", json!(POLICY)),
        ("max_batch", json!(maximum)),
        ("capacity_bootstrap_only", json!(request.capacity_bootstrap)),
        ("probe_capacity", json!(capacity)),
        ("requested_beam_width", json!(request.beam_width)),
        ("probe", json!(request.helper.sha256)),
        ("probe_protocol", json!(request.protocol)),
        ("artifact_files", json!(request.artifacts)),
        ("probe_manifest", request.probe_manifest.clone()),
        ("sample_policy", json!("center-splitmix64-128x16-v1")),
        (
            "graph_contract_sha256",
            json!(profile_cache::cache_key(&json!(request.graph))?),
        ),
        (
            "sample_sha256",
            json!(profile_cache::cache_key(&json!(samples))?),
        ),
    ] {
        fields.insert(key.into(), value);
    }
    let key = profile_cache::cache_key(&signature)?;
    let shared_batch = request
        .shared_profile
        .as_ref()
        .and_then(|v| v["inference"]["parent_batch"].as_u64())
        .filter(|n| !request.capacity_bootstrap && *n > 0 && *n <= cap);
    Ok(Prepared {
        signature,
        key,
        samples,
        capacity,
        cap,
        parents,
        world,
        devices,
        uuids,
        shared_batch,
    })
}
pub fn run(request: &Request, cancel: Arc<AtomicBool>) -> Result<Value> {
    let check_cancel = || {
        if cancel.load(Ordering::Acquire) {
            Err(Error::Message("inference calibration cancelled".into()))
        } else {
            Ok(())
        }
    };
    check_cancel()?;
    let prepared = prepare(request)?;
    request.helper.verify()?;
    artifacts::verify_all(&request.artifacts)?;
    check_cancel()?;
    let cache = request
        .options
        .cache_dir
        .join("profiles")
        .join(format!("{}.json", prepared.key));
    let free = telemetry::free_memory(&prepared.uuids);
    if let Some(hit) = profile_cache::read_inference(
        &cache,
        &prepared.signature,
        request.options.calibration_max_batch,
        free.as_ref(),
    )? {
        check_cancel()?;
        return Ok(hit);
    }
    let directory = request.run_dir.join("calibration");
    fs::create_dir(&directory)?;
    let inputs = directory.join("inputs");
    fs::create_dir(&inputs)?;
    let mut manifest = request.probe_manifest.clone();
    manifest["calibration_states"] = json!(prepared.samples);
    manifest["generators"] = json!(request.graph.generators);
    artifacts::atomic_json(
        &inputs.join(if request.ensemble {
            "ensemble.json"
        } else {
            "calibration.json"
        }),
        &manifest,
    )?;
    let started = Instant::now();
    let budget = Duration::try_from_secs_f64(if request.capacity_bootstrap {
        request.options.calibration_seconds.min(30.)
    } else {
        request.options.calibration_seconds
    })
    .map_err(|_| Error::Message("invalid calibration budget".into()))?;
    let mut pool = InferencePool::new(
        PoolConfig {
            command: vec![request.helper.path.to_string_lossy().into_owned()],
            inputs,
            capacity: prepared.capacity,
            parents: prepared.parents,
            world: prepared.world,
            environment: request.environment.clone(),
            cwd: request.run_dir.clone(),
            directory: directory.join("probes"),
            protocol: request.protocol,
        },
        budget,
    )?;
    pool.bind_cancellation(cancel.clone())?;
    let result = pool.tune(
        &TuneConfig {
            cap: prepared.cap,
            parents: prepared.parents,
            world: prepared.world,
            capacity_bootstrap: request.capacity_bootstrap,
            shared_batch: prepared.shared_batch,
            device_indices: prepared.devices,
            gpu_uuids: prepared.uuids,
        },
        budget,
    )?;
    check_cancel()?;
    request.helper.verify()?;
    artifacts::verify_all(&request.artifacts)?;
    check_cancel()?;
    let mut data = serde_json::to_value(result)?;
    data["signature"] = prepared.signature.clone();
    data["initial_free_memory_mib"] = json!(free);
    data["probe_protocol"] = json!(request.protocol);
    data["calibration_wall_seconds"] = json!(started.elapsed().as_secs_f64());
    data["shared_profile"] = if data["shared_profile_validated"] == true {
        json!(request.shared_profile)
    } else {
        Value::Null
    };
    profile_cache::write_inference(
        &cache,
        &data,
        &prepared.signature,
        request.options.calibration_max_batch,
    )?;
    artifacts::atomic_json(&directory.join("inference-selection.json"), &data)?;
    Ok(data)
}
