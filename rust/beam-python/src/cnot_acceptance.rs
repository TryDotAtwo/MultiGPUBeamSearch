//! Versioned, invocation-scoped CNOT backend evidence for strict acceptance.
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::PyResult;
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::OnceLock,
};

pub(crate) const EVENT_PREFIX: &str = "CNOT_ACCEPTANCE_EVENT ";
pub(crate) const EVENT_SCHEMA: &str = "org.trydotatwo.multigpubeamsearch.cnot.acceptance";
pub(crate) const EVENT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Identity {
    pub run_id: String,
    pub case_id: String,
    pub source_sha: String,
    pub wheel_sha256: String,
    pub gpu_uuids: Vec<String>,
    pub evidence_dir: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GpuObservation {
    pub rank: u32,
    pub local_rank: usize,
    pub device_ordinal: i32,
    pub gpu_uuid: String,
    pub context_handle: u64,
    pub thread_id: u64,
}

#[derive(Serialize)]
struct Event<'a> {
    schema: &'static str,
    version: u32,
    run_id: &'a str,
    case_id: &'a str,
    source_sha: &'a str,
    wheel_sha256: &'a str,
    kind: &'a str,
    rank: u32,
    world: u32,
    pid: u32,
    device: &'a str,
    gpu_uuid: &'a str,
    fields: &'a BTreeMap<String, String>,
}

fn valid_token(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_-".contains(&byte))
}

fn valid_sha(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn required_environment(name: &str) -> Result<String, String> {
    env::var(name).map_err(|_| format!("strict CNOT acceptance requires {name}"))
}

pub(crate) fn identity() -> Result<Option<Identity>, String> {
    let run_id = env::var("MGBS_CNOT_ACCEPTANCE_RUN_ID").ok();
    let case_id = env::var("MGBS_CNOT_ACCEPTANCE_CASE").ok();
    match (run_id, case_id) {
        (None, None) => Ok(None),
        (Some(run_id), Some(case_id)) => {
            if !valid_token(&run_id) || !valid_token(&case_id) {
                return Err("CNOT acceptance run and case IDs must be non-empty path-safe tokens".into());
            }
            let source_sha = required_environment("MGBS_CNOT_ACCEPTANCE_SOURCE_SHA")?.to_ascii_lowercase();
            let wheel_sha256 =
                required_environment("MGBS_CNOT_ACCEPTANCE_WHEEL_SHA256")?.to_ascii_lowercase();
            if !valid_sha(&source_sha, 40) || !valid_sha(&wheel_sha256, 64) {
                return Err("CNOT acceptance source and wheel identities must be SHA-1/SHA-256 hex".into());
            }
            let gpu_uuids = required_environment("MGBS_CNOT_ACCEPTANCE_GPU_UUIDS")?
                .split(',')
                .map(str::trim)
                .map(str::to_owned)
                .collect::<Vec<_>>();
            if gpu_uuids.is_empty()
                || gpu_uuids
                    .iter()
                    .any(|value| value.is_empty() || value.bytes().any(|byte| byte.is_ascii_whitespace()))
            {
                return Err("CNOT acceptance requires a comma-separated physical GPU UUID list".into());
            }
            if gpu_uuids.iter().collect::<BTreeSet<_>>().len() != gpu_uuids.len() {
                return Err("CNOT acceptance requires unique physical GPU UUIDs".into());
            }
            let evidence_dir = PathBuf::from(required_environment("CNOT_RUST_EVIDENCE")?);
            if evidence_dir.as_os_str().is_empty() {
                return Err("CNOT_RUST_EVIDENCE must identify the receipt root".into());
            }
            Ok(Some(Identity {
                run_id,
                case_id,
                source_sha,
                wheel_sha256,
                gpu_uuids,
                evidence_dir,
            }))
        }
        _ => Err("CNOT acceptance run ID and case ID must be set together".into()),
    }
}

fn valid_field_value(value: &str) -> bool {
    !value.is_empty()
        && !value.bytes().any(|byte| {
            byte.is_ascii_whitespace() || byte == b';' || byte == b'='
        })
}

pub(crate) fn contract_fields(contract: &str) -> PyResult<BTreeMap<String, String>> {
    let mut fields = BTreeMap::new();
    for entry in contract.split(';') {
        let (key, value) = entry.split_once('=').ok_or_else(|| {
            PyRuntimeError::new_err("strict CNOT backend contract has an invalid field")
        })?;
        if !valid_token(key) || !valid_field_value(value) || fields.contains_key(key) {
            return Err(PyRuntimeError::new_err(
                "strict CNOT backend contract has an invalid or duplicate field",
            ));
        }
        fields.insert(key.to_owned(), value.to_owned());
    }
    Ok(fields)
}

fn local_gpu_uuid(
    identity: &Identity,
    rank: u32,
    world: u32,
    observation: &GpuObservation,
) -> Result<String, String> {
    if observation.rank != rank || observation.local_rank >= identity.gpu_uuids.len()
        || identity.gpu_uuids.len() < world as usize || observation.device_ordinal < 0
        || observation.context_handle == 0 || observation.thread_id == 0 {
        return Err(format!("CNOT acceptance rank {rank} has an invalid admission-bound CUDA observation"));
    }
    let expected = &identity.gpu_uuids[observation.local_rank];
    if &observation.gpu_uuid != expected {
        return Err(format!("CNOT acceptance rank {rank} observed GPU {}, expected {expected}", observation.gpu_uuid));
    }
    Ok(observation.gpu_uuid.clone())
}

fn encode_event(
    identity: &Identity,
    kind: &str,
    rank: u32,
    world: u32,
    pid: u32,
    device: &str,
    gpu_uuid: &str,
    fields: &BTreeMap<String, String>,
) -> Result<String, String> {
    if !valid_token(kind)
        || world == 0
        || rank >= world
        || device.is_empty()
        || device.bytes().any(|byte| byte.is_ascii_whitespace())
        || gpu_uuid.is_empty()
        || gpu_uuid.bytes().any(|byte| byte.is_ascii_whitespace())
    {
        return Err("CNOT acceptance event has an invalid rank, world, device or GPU UUID".into());
    }
    serde_json::to_string(&Event {
        schema: EVENT_SCHEMA,
        version: EVENT_VERSION,
        run_id: &identity.run_id,
        case_id: &identity.case_id,
        source_sha: &identity.source_sha,
        wheel_sha256: &identity.wheel_sha256,
        kind,
        rank,
        world,
        pid,
        device,
        gpu_uuid,
        fields,
    })
    .map_err(|error| format!("serialize CNOT acceptance event: {error}"))
}

fn evidence_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

pub(crate) fn emit(
    kind: &str,
    rank: u32,
    world: u32,
    pid: u32,
    device: &str,
    fields: &BTreeMap<String, String>,
    observation: Option<&GpuObservation>,
) -> PyResult<()> {
    let identity = identity().map_err(PyRuntimeError::new_err)?;
    let Some(identity) = identity else {
        return Ok(());
    };
    let observation = observation.ok_or_else(|| PyRuntimeError::new_err(
        "CNOT acceptance requires an admission-bound CUDA observation"
    ))?;
    let gpu_uuid = local_gpu_uuid(&identity, rank, world, observation)
        .map_err(PyRuntimeError::new_err)?;
    let payload = encode_event(&identity, kind, rank, world, pid, device, &gpu_uuid, fields)
        .map_err(PyRuntimeError::new_err)?;

    let receipt_dir = identity.evidence_dir.join(&identity.run_id).join(&identity.case_id);
    fs::create_dir_all(&receipt_dir).map_err(|error| {
        PyRuntimeError::new_err(format!("cannot create CNOT acceptance receipt directory: {error}"))
    })?;
    let receipt_path = receipt_dir.join(format!("rank-{rank}-pid-{pid}.jsonl"));
    let _guard = evidence_lock().lock().map_err(|_| {
        PyRuntimeError::new_err("CNOT acceptance receipt writer lock was poisoned")
    })?;
    let mut receipt = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&receipt_path)
        .map_err(|error| {
            PyRuntimeError::new_err(format!("cannot open CNOT acceptance rank receipt: {error}"))
        })?;
    writeln!(receipt, "{payload}").map_err(|error| {
        PyRuntimeError::new_err(format!("cannot append CNOT acceptance rank receipt: {error}"))
    })?;
    eprintln!("{EVENT_PREFIX}{payload}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_identity() -> Identity {
        Identity {
            run_id: "attempt-17".into(),
            case_id: "trained-paired-1-gpu".into(),
            source_sha: "a".repeat(40),
            wheel_sha256: "b".repeat(64),
            gpu_uuids: vec!["GPU-0001".into()],
            evidence_dir: PathBuf::from("unused"),
        }
    }

    #[test]
    fn admission_bound_gpu_observation_must_match_its_rank_and_uuid() {
        let identity = fixture_identity();
        let observation = GpuObservation {
            rank: 0, local_rank: 0, device_ordinal: 1, gpu_uuid: "GPU-0001".into(),
            context_handle: 0x1234, thread_id: 7,
        };
        assert_eq!(local_gpu_uuid(&identity, 0, 1, &observation).unwrap(), "GPU-0001");
        assert!(local_gpu_uuid(&identity, 1, 1, &observation).is_err());
        let mut wrong_uuid = observation.clone();
        wrong_uuid.gpu_uuid = "GPU-OTHER".into();
        assert!(local_gpu_uuid(&identity, 0, 1, &wrong_uuid).is_err());
        let mut missing_context = observation;
        missing_context.context_handle = 0;
        assert!(local_gpu_uuid(&identity, 0, 1, &missing_context).is_err());
    }

    #[test]
    fn rust_serde_record_matches_the_shared_jsonl_producer_fixture() {
        let mut fields = BTreeMap::new();
        fields.insert("strict".into(), "1".into());
        fields.insert("scorer".into(), "rust-tqp".into());
        fields.insert("domain_abi".into(), "1".into());
        fields.insert("device_abi".into(), "1".into());
        fields.insert(
            "capabilities".into(),
            "owners,expansion,full_matrix_distinct".into(),
        );
        fields.insert("scorer_abi".into(), "1".into());
        fields.insert("domain_sha256".into(), "f".repeat(64));
        fields.insert("checkpoint_sha256".into(), "c".repeat(64));
        fields.insert("search_manifest_sha256".into(), "d".repeat(64));
        fields.insert("scorer_library_sha256".into(), "e".repeat(64));
        let event = encode_event(
            &fixture_identity(),
            "backend_admission",
            0,
            1,
            4000,
            "cuda:0",
            "GPU-0001",
            &fields,
        )
        .unwrap();
        let expected = include_str!("../../../tools/fixtures/cnot_acceptance_event_v1.jsonl")
            .trim()
            .strip_prefix(EVENT_PREFIX)
            .unwrap();
        assert_eq!(event, expected);
    }

    #[test]
    fn contract_semicolons_become_independent_fields_and_keep_capabilities() {
        let fields = contract_fields(
            "strict=1;domain_abi=1;device_abi=1;capabilities=owners,expansion,full_matrix_distinct;domain_sha256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        )
        .unwrap();
        assert_eq!(fields["domain_abi"], "1");
        assert_eq!(fields["device_abi"], "1");
        assert_eq!(fields["capabilities"], "owners,expansion,full_matrix_distinct");
        assert_eq!(fields["strict"], "1");
    }

    #[test]
    fn contract_rejects_duplicate_and_empty_fields() {
        assert!(contract_fields("abi=1;abi=2").is_err());
        assert!(contract_fields("abi=").is_err());
    }
}
