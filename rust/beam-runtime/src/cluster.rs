//! Externally scheduled rank supervision. CUDA and NCCL remain in the child.
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

const MAX_WORLD_SIZE: u64 = 128;

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanRequest {
    pub gpus_per_node: Vec<u32>,
    pub beam_width: u64,
    pub shard_count: u32,
    pub batch_alignment: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PlannedRank {
    pub global_rank: u64,
    pub node_rank: u64,
    pub local_rank: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ClusterPlan {
    pub ranks: Vec<PlannedRank>,
    pub requested_beam: u64,
    pub effective_beam: u64,
    pub alignment: u64,
    pub local_frontier_capacity: u64,
    pub shard_count: u32,
    pub hardware_verified: bool,
}

/// Preserve the public cluster planner's existing integer rules in beam-runtime.
pub fn plan_cluster(request: &PlanRequest) -> Result<ClusterPlan> {
    if request.gpus_per_node.is_empty() || request.gpus_per_node.contains(&0) {
        return Err(ClusterError::Invalid(
            "each node must have a positive GPU count".into(),
        ));
    }
    let world = request
        .gpus_per_node
        .iter()
        .try_fold(0u64, |total, &count| total.checked_add(u64::from(count)))
        .ok_or_else(|| ClusterError::Invalid("native cluster world size overflows".into()))?;
    if world > MAX_WORLD_SIZE {
        return Err(ClusterError::Invalid(format!(
            "native final exchange supports at most {MAX_WORLD_SIZE} ranks"
        )));
    }
    if request.beam_width == 0 {
        return Err(ClusterError::Invalid(
            "beam_width must be a positive uint64".into(),
        ));
    }
    if request.shard_count == 0 {
        return Err(ClusterError::Invalid(
            "shard_count must be a positive uint32".into(),
        ));
    }
    if request.batch_alignment == 0 {
        return Err(ClusterError::Invalid(
            "batch_alignment must be a positive uint32".into(),
        ));
    }
    let alignment =
        u128::from(world) * u128::from(request.shard_count) * u128::from(request.batch_alignment);
    let beam = u128::from(request.beam_width);
    let capacity = (beam + u128::from(world) - 1) / u128::from(world);
    if alignment >= (1u128 << 64) || capacity >= (1u128 << 32) {
        return Err(ClusterError::Invalid(
            "native alignment or rank-local frontier index overflows".into(),
        ));
    }

    let mut ranks = Vec::with_capacity(world as usize);
    for (node_rank, &count) in request.gpus_per_node.iter().enumerate() {
        for local_rank in 0..u64::from(count) {
            ranks.push(PlannedRank {
                global_rank: ranks.len() as u64,
                node_rank: node_rank as u64,
                local_rank,
            });
        }
    }

    Ok(ClusterPlan {
        ranks,
        requested_beam: request.beam_width,
        effective_beam: request.beam_width,
        alignment: alignment as u64,
        local_frontier_capacity: capacity as u64,
        shard_count: request.shard_count,
        hardware_verified: false,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum ClusterError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Runtime(String),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
type Result<T> = std::result::Result<T, ClusterError>;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ScheduledRank {
    pub global_rank: u64,
    pub node_rank: u64,
    pub local_rank: u64,
    pub world_size: u64,
    pub local_world_size: u64,
}

fn number(env: &BTreeMap<String, String>, name: &str) -> Result<u64> {
    let value = env.get(name).map(String::as_str).unwrap_or("");
    if value.is_empty() || value.len() > 10 || !value.bytes().all(|c| c.is_ascii_digit()) {
        return Err(ClusterError::Invalid(format!(
            "missing or invalid scheduler variable {name}"
        )));
    }
    value
        .parse()
        .map_err(|_| ClusterError::Invalid(format!("missing or invalid scheduler variable {name}")))
}

fn slurm_counts(value: &str) -> Result<Vec<u64>> {
    fn positive(text: &str) -> Option<u64> {
        if text.is_empty()
            || text.len() > 3
            || text.starts_with('0')
            || !text.bytes().all(|c| c.is_ascii_digit())
        {
            return None;
        }
        text.parse().ok()
    }
    let mut counts = Vec::new();
    for part in value.split(',') {
        let (count, repeats) = match part.split_once("(x") {
            Some((count, rest)) => (positive(count), rest.strip_suffix(')').and_then(positive)),
            None => (positive(part), Some(1)),
        };
        let (count, repeats) = match (count, repeats) {
            (Some(count), Some(repeats)) => (count, repeats),
            _ => {
                return Err(ClusterError::Invalid(
                    "invalid Slurm tasks-per-node list".into(),
                ))
            }
        };
        if counts.len() as u64 + repeats > MAX_WORLD_SIZE {
            return Err(ClusterError::Invalid("invalid Slurm node count".into()));
        }
        counts.extend(std::iter::repeat_n(count, repeats as usize));
    }
    Ok(counts)
}

pub fn scheduler_rank(env: &BTreeMap<String, String>) -> Result<ScheduledRank> {
    let (world, rank, local_size, local, node);
    let mut rank_offset = None;
    if env.contains_key("TORCHELASTIC_RUN_ID") || !env.contains_key("SLURM_PROCID") {
        world = number(env, "WORLD_SIZE")?;
        rank = number(env, "RANK")?;
        local_size = number(env, "LOCAL_WORLD_SIZE")?;
        local = number(env, "LOCAL_RANK")?;
        node = number(env, "GROUP_RANK")?;
        if local_size == 0 || world % local_size != 0 || node >= world / local_size {
            return Err(ClusterError::Invalid(
                "invalid torchrun node or local group size".into(),
            ));
        }
    } else {
        world = number(
            env,
            if env.contains_key("SLURM_STEP_NUM_TASKS") {
                "SLURM_STEP_NUM_TASKS"
            } else {
                "SLURM_NTASKS"
            },
        )?;
        rank = number(env, "SLURM_PROCID")?;
        local = number(env, "SLURM_LOCALID")?;
        node = number(env, "SLURM_NODEID")?;
        let counts = slurm_counts(
            env.get("SLURM_STEP_TASKS_PER_NODE")
                .or_else(|| env.get("SLURM_TASKS_PER_NODE"))
                .map(String::as_str)
                .unwrap_or(""),
        )?;
        if counts.iter().sum::<u64>() != world || node >= counts.len() as u64 {
            return Err(ClusterError::Invalid(
                "Slurm node counts disagree with the current step".into(),
            ));
        }
        local_size = counts[node as usize];
        rank_offset = Some(counts[..node as usize].iter().sum::<u64>());
    }
    if !(1..=MAX_WORLD_SIZE).contains(&world) || rank >= world || local >= local_size {
        return Err(ClusterError::Invalid(format!(
            "invalid rank or world size (maximum {MAX_WORLD_SIZE})"
        )));
    }
    if let Some(offset) = rank_offset {
        if rank != offset + local {
            return Err(ClusterError::Invalid(
                "Slurm rank does not match node/local assignment".into(),
            ));
        }
    }
    Ok(ScheduledRank {
        global_rank: rank,
        node_rank: node,
        local_rank: local,
        world_size: world,
        local_world_size: local_size,
    })
}

/// Apply the existing scheduler-rank to process-local CUDA ordinal mapping.
pub fn cuda_ordinal(local_rank: i64, visible_devices: u64) -> Result<i64> {
    if visible_devices == 0 {
        return Err(ClusterError::Invalid("no visible CUDA devices".into()));
    }
    if visible_devices == 1 {
        return Ok(0);
    }
    if i128::from(local_rank) >= i128::from(visible_devices) {
        return Err(ClusterError::Invalid(
            "LOCAL_RANK exceeds the visible CUDA devices".into(),
        ));
    }
    Ok(local_rank)
}

/// Map selected logical CUDA indices through a scheduler-provided visibility list.
pub fn cuda_visibility(devices: &[u32], inherited: Option<&str>) -> Result<String> {
    let Some(inherited) = inherited else {
        return Ok(devices
            .iter()
            .map(|device| device.to_string())
            .collect::<Vec<_>>()
            .join(","));
    };
    let tokens: Vec<&str> = inherited.split(',').map(str::trim).collect();
    let mut mapped = Vec::with_capacity(devices.len());
    for &device in devices {
        let index = usize::try_from(device).map_err(|_| {
            ClusterError::Invalid(
                "cannot map selected devices through inherited CUDA_VISIBLE_DEVICES".into(),
            )
        })?;
        let token = tokens.get(index).copied().unwrap_or("");
        if token.is_empty() || token == "-1" {
            return Err(ClusterError::Invalid(
                "cannot map selected devices through inherited CUDA_VISIBLE_DEVICES".into(),
            ));
        }
        mapped.push(token);
    }
    Ok(mapped.join(","))
}

const CHILD_ENV_FILTER_PREFIXES: [&str; 12] = [
    "BEAM_",
    "TORCHELASTIC_",
    "TORCHRUN_",
    "CUDA_",
    "NCCL_",
    "PUBLISH_",
    "RESULTS_INGEST_",
    "SHARD_",
    "STREAM1_",
    "STREAM3_",
    "STREAM4_",
    "GLOBAL_",
];

const CHILD_ENV_FILTER_KEYS: [&str; 11] = [
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NODE_RANK",
    "OMP_NUM_THREADS",
];

fn child_env_key_is_filtered(key: &str) -> bool {
    CHILD_ENV_FILTER_KEYS.contains(&key)
        || CHILD_ENV_FILTER_PREFIXES
            .iter()
            .any(|prefix| key.starts_with(prefix))
}

/// Build the same isolated child environment used by the Python adapter.
pub fn child_environment(
    inherited: &BTreeMap<String, String>,
    devices: &[u32],
    nccl_id_file: &str,
) -> Result<BTreeMap<String, String>> {
    let mut environment: BTreeMap<String, String> = inherited
        .iter()
        .filter(|(key, _)| !child_env_key_is_filtered(key))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    environment.insert(
        "CUDA_VISIBLE_DEVICES".into(),
        cuda_visibility(
            devices,
            inherited.get("CUDA_VISIBLE_DEVICES").map(String::as_str),
        )?,
    );
    if let Some(order) = inherited.get("CUDA_DEVICE_ORDER") {
        environment.insert("CUDA_DEVICE_ORDER".into(), order.clone());
    }
    environment.insert("BEAM_NCCL_ID_FILE".into(), nccl_id_file.into());
    Ok(environment)
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchEnvironmentRequest {
    pub puzzle_info: String,
    pub test_csv: String,
    pub weight_dir: String,
    pub run_dir: String,
    pub effective_touch_bfs_radius: u32,
    pub touch_bfs_max_entries: u64,
}

/// Build the fixed permutation worker environment consumed by the native runner.
pub fn search_environment(request: &SearchEnvironmentRequest) -> BTreeMap<String, String> {
    let history_dir = Path::new(&request.run_dir)
        .join("history")
        .to_string_lossy()
        .into_owned();
    [
        ("BEAM_GENERATOR_PATH", request.puzzle_info.clone()),
        ("BEAM_PUZZLE_INFO_JSON", request.puzzle_info.clone()),
        ("BEAM_TEST_CSV", request.test_csv.clone()),
        ("BEAM_WEIGHT_DIR", request.weight_dir.clone()),
        ("BEAM_RUNTIME_CONFIG_MODE", "auto".into()),
        (
            "BEAM_SOLVED_NEIGHBORHOOD_RADIUS",
            request.effective_touch_bfs_radius.to_string(),
        ),
        (
            "BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES",
            request.touch_bfs_max_entries.to_string(),
        ),
        ("BEAM_STREAM2_SUFFIX_RADIUS", "0".into()),
        ("BEAM_SOLVE_BUCKET_MODE", "0".into()),
        ("BEAM_HISTORY_MODE", "disk".into()),
        ("BEAM_HISTORY_DIR", history_dir.clone()),
        ("BEAM_HISTORY_DISK_PATH", history_dir),
    ]
    .into_iter()
    .map(|(key, value)| (key.into(), value))
    .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MlpHostBindingIdentity {
    pub graph_hash: String,
    pub artifact_hash: String,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchModelEnvironmentRequest {
    pub model_backend: String,
    pub ensemble_blend_dir: Option<String>,
    pub ensemble_first_weights_dir: Option<String>,
    pub inference_backend: Option<String>,
    pub mlp_host_binding: Option<MlpHostBindingIdentity>,
}

/// Build fixed model-family variables for the permutation runner.
/// Python still owns framework objects and model preparation; the Rust runtime
/// receives only the prepared paths, route, and optional binding identity.
pub fn search_model_environment(
    request: &SearchModelEnvironmentRequest,
) -> Result<BTreeMap<String, String>> {
    if request.model_backend.is_empty() {
        return Err(ClusterError::Invalid(
            "model backend must be a nonempty string".into(),
        ));
    }
    if request
        .inference_backend
        .as_deref()
        .is_some_and(|backend| !matches!(backend, "libtorch" | "cutlass"))
    {
        return Err(ClusterError::Invalid(
            "unsupported model inference backend".into(),
        ));
    }

    let ensemble = request.model_backend == "ensemble";
    let mut environment = BTreeMap::new();
    match (
        ensemble,
        request.ensemble_blend_dir.as_deref(),
        request.ensemble_first_weights_dir.as_deref(),
    ) {
        (true, Some(blend_dir), Some(first_weights_dir))
            if !blend_dir.is_empty() && !first_weights_dir.is_empty() =>
        {
            environment.insert("BEAM_BLEND_DIR".into(), blend_dir.into());
            environment.insert("BEAM_WEIGHT_DIR".into(), first_weights_dir.into());
            if let Some(backend) = request.inference_backend.as_deref() {
                environment.insert("BEAM_ENSEMBLE_BACKEND".into(), backend.into());
            }
        }
        (false, None, None) => {}
        _ => {
            return Err(ClusterError::Invalid(
                "ensemble environment paths do not match the model family".into(),
            ))
        }
    }

    if request.inference_backend.as_deref() == Some("libtorch")
        || matches!(
            request.model_backend.as_str(),
            "piece_transformer" | "ensemble"
        )
    {
        environment.insert("BEAM_STREAM1_EXECUTOR".into(), "libtorch_eager".into());
    }

    if let Some(binding) = &request.mlp_host_binding {
        let valid_hash = |value: &str| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        };
        if request.model_backend != "mlp"
            || !valid_hash(&binding.graph_hash)
            || !valid_hash(&binding.artifact_hash)
        {
            return Err(ClusterError::Invalid(
                "MLP host binding requires an MLP model and SHA256 identities".into(),
            ));
        }
        environment.insert("BEAM_MLP_GRAPH_HASH".into(), binding.graph_hash.clone());
        environment.insert(
            "BEAM_MLP_ARTIFACT_HASH".into(),
            binding.artifact_hash.clone(),
        );
    }
    Ok(environment)
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchPolicyEnvironmentRequest {
    pub solution_mode: String,
    pub collect_extra_depths: u32,
    pub max_collected_solutions: u64,
}

/// Build the fixed solution collection policy passed to the permutation runner.
pub fn search_policy_environment(
    request: &SearchPolicyEnvironmentRequest,
) -> Result<BTreeMap<String, String>> {
    let solve_bucket_mode = match request.solution_mode.as_str() {
        "first" => "0",
        "collect" => "1",
        _ => {
            return Err(ClusterError::Invalid(
                "solution_mode must be first or collect".into(),
            ))
        }
    };
    if !(1..=1_000_000).contains(&request.max_collected_solutions) {
        return Err(ClusterError::Invalid(
            "max_collected_solutions must be between 1 and 1000000".into(),
        ));
    }

    Ok([
        ("BEAM_SOLVE_BUCKET_MODE", solve_bucket_mode.to_string()),
        (
            "BEAM_SOLVE_BUCKET_EXTRA_DEPTHS",
            request.collect_extra_depths.to_string(),
        ),
        (
            "BEAM_SOLVE_BUCKET_MAX_SOLUTIONS",
            request.max_collected_solutions.to_string(),
        ),
        (
            "BEAM_SOLVE_BUCKET_RESULT_TSV",
            "collected-solutions.tsv".into(),
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.into(), value))
    .collect())
}

pub struct Request {
    pub command: Vec<String>,
    /// Executable lookup and Python pathlib resolution preserve existing Python semantics.
    pub executable: Option<PathBuf>,
    pub shared_dir: PathBuf,
    pub run_id: Option<String>,
    pub timeout: f64,
    pub environment: BTreeMap<String, String>,
}

fn valid_run_id(id: &str) -> bool {
    let mut bytes = id.bytes();
    id.len() <= 128
        && id != "none"
        && bytes.next().is_some_and(|c| c.is_ascii_alphanumeric())
        && bytes.all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
}

struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(None)) {
            return;
        }
        #[cfg(unix)]
        let _ = beam_native_sys::terminate_process(self.0.id());
        #[cfg(not(unix))]
        let _ = self.0.kill();
        let began = Instant::now();
        while began.elapsed() < Duration::from_secs(2) {
            if !matches!(self.0.try_wait(), Ok(None)) {
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Own a single scheduler-assigned native child until it exits or is cancelled.
pub fn run(mut request: Request, cancel: &AtomicBool) -> Result<PathBuf> {
    let rank = scheduler_rank(&request.environment)?;
    if request.command.is_empty() || !request.timeout.is_finite() || request.timeout <= 0.0 {
        return Err(ClusterError::Invalid(
            "a native argument vector and a positive finite timeout are required".into(),
        ));
    }
    let run_id = request
        .run_id
        .or_else(|| {
            request
                .environment
                .get("TORCHELASTIC_RUN_ID")
                .cloned()
                .filter(|s| !s.is_empty())
        })
        .or_else(|| {
            match (
                request.environment.get("SLURM_JOB_ID"),
                request.environment.get("SLURM_STEP_ID"),
            ) {
                (Some(job), Some(step)) if !job.is_empty() && !step.is_empty() => {
                    Some(format!("slurm-{job}-{step}"))
                }
                _ => None,
            }
        })
        .filter(|s| valid_run_id(s))
        .ok_or_else(|| {
            ClusterError::Invalid("supply a unique --run-id shared by all nodes".into())
        })?;
    let attempt = if request
        .environment
        .contains_key("TORCHELASTIC_RESTART_COUNT")
    {
        number(&request.environment, "TORCHELASTIC_RESTART_COUNT")?
    } else {
        0
    };
    let root = request
        .shared_dir
        .join(&run_id)
        .join(format!("attempt-{attempt}"));
    fs::create_dir_all(&root)?;
    let directory = root.join(format!("rank-{}", rank.global_rank));
    fs::create_dir(&directory)?;
    let failed = root.join("FAILED");
    let mut owned_child: Option<OwnedChild> = None;
    let result = (|| -> Result<PathBuf> {
        fs::create_dir(root.join(format!("node-{}-local-{}", rank.node_rank, rank.local_rank)))?;
        if failed.exists() {
            return Err(ClusterError::Runtime(format!(
                "a peer failed; see {}",
                failed.display()
            )));
        }
        for (name, value) in [
            ("RANK", rank.global_rank.to_string()),
            ("WORLD_SIZE", rank.world_size.to_string()),
            ("GROUP_RANK", rank.node_rank.to_string()),
            ("LOCAL_RANK", rank.local_rank.to_string()),
            ("LOCAL_WORLD_SIZE", rank.local_world_size.to_string()),
            (
                "BEAM_NCCL_ID_FILE",
                root.join("nccl-id.bin").to_string_lossy().into_owned(),
            ),
            ("BEAM_NCCL_RUN_ID", run_id.clone()),
            ("BEAM_CLUSTER_LAUNCH", "1".into()),
        ] {
            request.environment.insert(name.into(), value);
        }
        request
            .environment
            .entry("BEAM_HISTORY_MODE".into())
            .or_insert_with(|| "disk".into());
        request
            .environment
            .entry("BEAM_HISTORY_DIR".into())
            .or_insert_with(|| directory.join("history").to_string_lossy().into_owned());
        let executable = request.executable.as_ref().ok_or_else(|| {
            ClusterError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("cannot find native executable: {}", request.command[0]),
            ))
        })?;
        request.command[0] = executable.to_string_lossy().into_owned();
        let launch = serde_json::json!({"argv":request.command,"rank":rank.global_rank,"world_size":rank.world_size,
            "node":rank.node_rank,"local_rank":rank.local_rank,"local_world_size":rank.local_world_size,
            "cuda_visible_devices":request.environment.get("CUDA_VISIBLE_DEVICES")});
        let mut manifest = File::create(directory.join("launch.json"))?;
        serde_json::to_writer_pretty(&mut manifest, &launch)?;
        writeln!(manifest)?;
        let log_path = directory.join("native.log");
        let log = File::create(&log_path)?;
        let started = Instant::now();
        let child = Command::new(executable)
            .args(&request.command[1..])
            .env_clear()
            .envs(&request.environment)
            .current_dir(&directory)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log))
            .spawn()?;
        owned_child = Some(OwnedChild(child));
        loop {
            if let Some(status) = owned_child.as_mut().expect("owned child").0.try_wait()? {
                if status.success() {
                    return Ok(directory.clone());
                }
                #[cfg(unix)]
                let code = status.code().unwrap_or_else(|| {
                    use std::os::unix::process::ExitStatusExt;
                    -status.signal().unwrap_or(0)
                });
                #[cfg(not(unix))]
                let code = status.code().unwrap_or(-1);
                return Err(ClusterError::Runtime(format!(
                    "native rank {} exited {code}; see {}",
                    rank.global_rank,
                    log_path.display()
                )));
            }
            if failed.exists() {
                return Err(ClusterError::Runtime(format!(
                    "a peer failed; see {}",
                    failed.display()
                )));
            }
            if cancel.load(Ordering::Acquire) {
                return Err(ClusterError::Runtime("native rank cancelled".into()));
            }
            let remaining = request.timeout - started.elapsed().as_secs_f64();
            if remaining <= 0.0 {
                return Err(ClusterError::Runtime(format!(
                    "native rank {} timed out; see {}",
                    rank.global_rank,
                    log_path.display()
                )));
            }
            thread::sleep(Duration::from_secs_f64(remaining.min(0.2)));
        }
    })();
    if let Err(error) = &result {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&failed)
        {
            Ok(mut file) => writeln!(file, "rank {}: {error}", rank.global_rank)?,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
    }
    result
}
