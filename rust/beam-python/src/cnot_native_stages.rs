//! Rank-local CNOT stage ownership for the Rust search controller.
//!
//! Tensor storage remains owned by PyTorch/CUDA. Existing ATen operators and
//! rank collectives are invoked on the caller's thread in their former order.
use crate::{
    cnot::PyCnotRankedRequest, cnot_native_batch, cnot_native_checked::checked_v2 as checked,
    cnot_native_exchange, cnot_scorer_dispatch,
};
use beam_runtime::cnot_search::{self, Backend, Edge, Goal, Options, RankedCnotRequest, Round};
use pyo3::{
    exceptions::{PyOverflowError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyTuple},
};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

#[pyclass(name = "_CnotBackendAdmission", unsendable)]
pub(crate) struct PyCnotBackendAdmission {
    pub(crate) strict: bool,
    device_domain: Option<beam_native_sys::cnot_device_bridge::DeviceDomain>,
    device_domain_sha256: Option<String>,
    gpu_observation: Option<crate::cnot_acceptance::GpuObservation>,
    stream5_library_sha256: Option<String>,
    stream5_transport: Option<Py<PyAny>>,
}

#[pymethods]
impl PyCnotBackendAdmission {
    #[getter]
    fn strict(&self) -> bool {
        self.strict
    }
}

fn strict_backend_enabled() -> bool {
    matches!(
        std::env::var("MGBS_CNOT_STRICT_BACKEND").as_deref(),
        Ok("1")
    )
}

pub(crate) fn observe_acceptance_gpu(
    py: Python<'_>,
    world: &Bound<'_, PyAny>,
    device: &Bound<'_, PyAny>,
) -> PyResult<Option<crate::cnot_acceptance::GpuObservation>> {
    if crate::cnot_acceptance::identity()
        .map_err(PyRuntimeError::new_err)?
        .is_none()
    {
        return Ok(None);
    }
    let rank: u32 = world.getattr("rank")?.extract()?;
    let size: u32 = world.getattr("size")?.extract()?;
    let module = py.import("multigpubeamsearch.cnot_acceptance")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("device", device)?;
    let observation =
        module.call_method("gpu_observation_for_rank", (rank, size), Some(&kwargs))?;
    let local_rank = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(rank as usize);
    Ok(Some(crate::cnot_acceptance::GpuObservation {
        rank,
        local_rank,
        device_ordinal: observation.getattr("device_ordinal")?.extract()?,
        gpu_uuid: observation.getattr("gpu_uuid")?.extract()?,
        context_handle: observation.getattr("context_handle")?.extract()?,
        thread_id: observation.getattr("thread_id")?.extract()?,
    }))
}

pub(crate) fn strict_backend_mode(world: &Bound<'_, PyAny>) -> PyResult<bool> {
    let requested = strict_backend_enabled();
    if world.getattr("size")?.extract::<usize>()? == 1 {
        return Ok(requested);
    }
    let all_strict: bool = world
        .call_method1("all_ranks_support", (requested,))?
        .extract()?;
    let all_compatible: bool = world
        .call_method1("all_ranks_support", (!requested,))?
        .extract()?;
    if all_strict {
        return Ok(true);
    }
    if all_compatible {
        return Ok(false);
    }
    Err(PyRuntimeError::new_err(
        "strict CNOT acceptance activation differs between ranks",
    ))
}

pub(crate) fn file_sha256(path: &std::path::Path) -> PyResult<String> {
    let contents = fs::read(path).map_err(|error| {
        PyRuntimeError::new_err(format!("cannot read CNOT DeviceDomain library: {error}"))
    })?;
    let digest = Sha256::digest(contents);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(encoded)
}

fn load_device_domain_admission(
    path: &std::path::Path,
    device: i32,
    n: i32,
    world: i32,
) -> PyResult<(beam_native_sys::cnot_device_bridge::DeviceDomain, String)> {
    let domain_sha = file_sha256(path)?;
    let domain = beam_native_sys::cnot_device_bridge::DeviceDomain::load(path, device, n, world)?;
    Ok((domain, domain_sha))
}

fn prepare_device_domain_checked<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    prepare: impl FnOnce() -> PyResult<(beam_native_sys::cnot_device_bridge::DeviceDomain, String)>,
    fail_rank: Option<i64>,
) -> PyResult<(beam_native_sys::cnot_device_bridge::DeviceDomain, String)> {
    let mut prepared = None;
    let local = (|| -> PyResult<Bound<'py, PyAny>> {
        let value = prepare()?;
        let rank: i64 = world.getattr("rank")?.extract()?;
        if fail_rank == Some(rank) {
            return Err(PyRuntimeError::new_err(
                "injected CNOT DeviceDomain preparation failure",
            ));
        }
        prepared = Some(value);
        Ok(py.None().into_bound(py))
    })();
    checked(py, world, "prepare admitted CNOT DeviceDomain", local)?;
    prepared.ok_or_else(|| {
        PyRuntimeError::new_err("CNOT DeviceDomain preparation returned no admitted handle")
    })
}

fn device_domain_contract(
    path: &std::path::Path,
    device: i32,
    n: i32,
    world: i32,
) -> PyResult<String> {
    let (_domain, domain_sha) = load_device_domain_admission(path, device, n, world)?;
    Ok(format!(
        "domain_abi=1;device_abi=1;capabilities=owners,expansion,full_matrix_distinct;domain_sha256={domain_sha}"
    ))
}

fn emit_device_dispatch(
    world: &Bound<'_, PyAny>,
    rank: u32,
    size: u32,
    strict: bool,
    sha256: Option<&str>,
    operation: &str,
    rows: usize,
    observation: Option<&crate::cnot_acceptance::GpuObservation>,
) -> PyResult<()> {
    if !strict {
        return Ok(());
    }
    let device = world.getattr("device")?.str()?.to_str()?.to_owned();
    let fields = [
        ("implementation".to_owned(), "DeviceDomain".to_owned()),
        ("operation".to_owned(), operation.to_owned()),
        ("rows".to_owned(), rows.to_string()),
        ("abi".to_owned(), "1".to_owned()),
        ("device_abi".to_owned(), "1".to_owned()),
        (
            "library_sha256".to_owned(),
            sha256.unwrap_or("unknown").to_owned(),
        ),
    ]
    .into_iter()
    .collect();
    crate::cnot_acceptance::emit(
        "device_domain_dispatch",
        rank,
        size,
        std::process::id(),
        &device,
        &fields,
        observation,
    )?;
    Ok(())
}

fn backend_admission_contract(
    strict: bool,
    path: Option<&std::path::Path>,
    is_cuda: bool,
    device: i32,
    n: i32,
    world: i32,
) -> PyResult<String> {
    if !strict {
        return Ok("strict=0".to_owned());
    }
    if !is_cuda {
        return Err(PyRuntimeError::new_err(
            "strict CNOT acceptance requires CUDA DeviceDomain inputs",
        ));
    }
    let path = path.ok_or_else(|| {
        PyRuntimeError::new_err("strict CNOT acceptance requires MGBS_CNOT_OWNED_LIBRARY")
    })?;
    Ok(format!(
        "strict=1;{}",
        device_domain_contract(path, device, n, world)?
    ))
}

#[pyfunction(name = "_probe_cnot_backend_admission")]
fn probe_cnot_backend_admission(
    strict: bool,
    path: Option<PathBuf>,
    is_cuda: bool,
    device: i32,
    n: i32,
    world: i32,
) -> PyResult<String> {
    backend_admission_contract(strict, path.as_deref(), is_cuda, device, n, world)
}

fn require_backend_consensus(world: &Bound<'_, PyAny>, contract: &str) -> PyResult<()> {
    // Rank-local failures can make the textual contracts differ in length.
    // Compare a fixed-size digest so Gloo never enters all_gather with
    // mismatched tensor sizes.
    let digest = Sha256::digest(contract.as_bytes()).to_vec();
    world.call_method1("require_engine_consensus", (digest,))?;
    Ok(())
}

#[pyfunction(name = "_require_cnot_backend_consensus")]
fn require_cnot_backend_consensus(world: Bound<'_, PyAny>, contract: &str) -> PyResult<()> {
    require_backend_consensus(&world, contract)
}

/// Validate required trained-search providers before transport startup. All
/// local failures are voted over Gloo, then the rank contracts are compared
/// over that same control group before any search data plane can begin.
pub(crate) fn preflight_strict_backend<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    root: &Bound<'py, PyAny>,
    scorer: &Bound<'py, PyAny>,
    use_rust_controller: bool,
) -> PyResult<Py<PyCnotBackendAdmission>> {
    let strict = strict_backend_mode(world)?;
    if !strict {
        return Ok(Py::new(
            py,
            PyCnotBackendAdmission {
                strict: false,
                device_domain: None,
                device_domain_sha256: None,
                gpu_observation: None,
                stream5_library_sha256: None,
                stream5_transport: None,
            },
        )?);
    }
    let (device_domain, domain_sha256) = prepare_device_domain_checked(
        py,
        world,
        || {
            if !use_rust_controller {
                return Err(PyRuntimeError::new_err(
                    "strict CNOT acceptance requires the Rust search controller",
                ));
            }
            if !root.getattr("is_cuda")?.extract::<bool>()? {
                return Err(PyRuntimeError::new_err(
                    "strict CNOT acceptance requires CUDA DeviceDomain inputs",
                ));
            }
            let path = std::env::var_os("MGBS_CNOT_OWNED_LIBRARY")
                .map(PathBuf::from)
                .ok_or_else(|| {
                    PyRuntimeError::new_err(
                        "strict CNOT acceptance requires MGBS_CNOT_OWNED_LIBRARY",
                    )
                })?;
            let device: Option<i32> = root.getattr("device")?.getattr("index")?.extract()?;
            let device = device.ok_or_else(|| {
                PyRuntimeError::new_err("strict CNOT acceptance requires an explicit CUDA device")
            })?;
            let n: i32 = root.call_method1("size", (1,))?.extract()?;
            let world_size: i32 = world.getattr("size")?.extract()?;
            load_device_domain_admission(&path, device, n, world_size)
        },
        None,
    )?;
    let mut admitted_backend = None;
    let local = (|| -> PyResult<Bound<'py, PyAny>> {
        let device_contract = format!(
            "strict=1;domain_abi=1;device_abi=1;capabilities=owners,expansion,full_matrix_distinct;domain_sha256={domain_sha256}"
        );
        let root_device = root.getattr("device")?;
        let gpu_observation = observe_acceptance_gpu(py, world, &root_device)?;

        if !scorer.is_instance_of::<crate::cnot_tqp_scorer::PyTqpScorer>() {
            return Err(PyRuntimeError::new_err(
                "strict CNOT acceptance requires the Rust TQP checkpoint scorer",
            ));
        }
        let metadata = scorer.getattr("metadata")?;
        let checkpoint: String =
            metadata
                .get_item("checkpoint_sha256")?
                .extract()
                .map_err(|_| {
                    PyRuntimeError::new_err("Rust TQP scorer metadata lacks checkpoint_sha256")
                })?;
        let search_manifest: String = metadata
            .get_item("search_manifest_sha256")?
            .extract()
            .map_err(|_| {
                PyRuntimeError::new_err("Rust TQP scorer metadata lacks search_manifest_sha256")
            })?;
        let scorer_library: String = metadata
            .get_item("scorer_library_sha256")?
            .extract()
            .map_err(|_| {
                PyRuntimeError::new_err("Rust TQP scorer metadata lacks scorer_library_sha256")
            })?;
        let expected = std::env::var("MGBS_CNOT_EXPECTED_CHECKPOINT_SHA256")
            .map_err(|_| {
                PyRuntimeError::new_err(
                    "strict CNOT acceptance requires MGBS_CNOT_EXPECTED_CHECKPOINT_SHA256",
                )
            })?
            .to_ascii_lowercase();
        if expected.len() != 64
            || !expected.bytes().all(|byte| byte.is_ascii_hexdigit())
            || checkpoint.to_ascii_lowercase() != expected
        {
            return Err(PyRuntimeError::new_err(format!(
                "strict CNOT checkpoint mismatch: expected={expected} actual={checkpoint}"
            )));
        }
        let expected_manifest = std::env::var("MGBS_CNOT_EXPECTED_SEARCH_MANIFEST_SHA256")
            .map_err(|_| {
                PyRuntimeError::new_err(
                    "strict CNOT acceptance requires MGBS_CNOT_EXPECTED_SEARCH_MANIFEST_SHA256",
                )
            })?
            .to_ascii_lowercase();
        if expected_manifest.len() != 64
            || !expected_manifest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || search_manifest.to_ascii_lowercase() != expected_manifest
        {
            return Err(PyRuntimeError::new_err(format!(
                "strict CNOT search manifest mismatch: expected={expected_manifest} actual={search_manifest}"
            )));
        }
        let expected_scorer_library = std::env::var("MGBS_CNOT_EXPECTED_TQP_LIBRARY_SHA256")
            .map_err(|_| {
                PyRuntimeError::new_err(
                    "strict CNOT acceptance requires MGBS_CNOT_EXPECTED_TQP_LIBRARY_SHA256",
                )
            })?
            .to_ascii_lowercase();
        if expected_scorer_library.len() != 64
            || !expected_scorer_library
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || scorer_library.to_ascii_lowercase() != expected_scorer_library
        {
            return Err(PyRuntimeError::new_err(format!(
                "strict CNOT scorer library mismatch: expected={expected_scorer_library} actual={scorer_library}"
            )));
        }
        let contract = format!(
            "{device_contract};scorer=rust-tqp;scorer_abi=1;checkpoint_sha256={expected};search_manifest_sha256={expected_manifest};scorer_library_sha256={expected_scorer_library}"
        );
        admitted_backend = Some(Py::new(
            py,
            PyCnotBackendAdmission {
                strict: true,
                device_domain: Some(device_domain),
                device_domain_sha256: Some(domain_sha256),
                gpu_observation: gpu_observation.clone(),
                stream5_library_sha256: None,
                stream5_transport: None,
            },
        )?);
        Ok(pyo3::types::PyString::new(py, &contract).into_any())
    })();
    let contract = checked(py, world, "admit strict CNOT backends", local)?;
    let contract: String = contract.extract()?;
    let admitted = admitted_backend.ok_or_else(|| {
        PyRuntimeError::new_err("strict CNOT admission returned no retained backend handle")
    })?;
    let gpu_observation = admitted.borrow(py).gpu_observation.clone();
    let consensus_contract = (|| -> PyResult<Bound<'py, PyAny>> {
        let identity = crate::cnot_acceptance::identity().map_err(PyRuntimeError::new_err)?;
        let value = if let Some(identity) = identity {
            format!(
                "{contract};acceptance_run_id={};acceptance_case={};acceptance_source_sha={};acceptance_wheel_sha256={}",
                identity.run_id, identity.case_id, identity.source_sha, identity.wheel_sha256
            )
        } else {
            contract.clone()
        };
        Ok(pyo3::types::PyString::new(py, &value).into_any())
    })();
    let consensus_contract = checked(
        py,
        world,
        "prepare strict CNOT acceptance identity",
        consensus_contract,
    )?;
    let consensus_contract: String = consensus_contract.extract()?;
    require_backend_consensus(world, &consensus_contract)?;
    let receipt_write = (|| -> PyResult<Bound<'py, PyAny>> {
        let rank: u32 = world.getattr("rank")?.extract()?;
        let size: u32 = world.getattr("size")?.extract()?;
        let pid = std::process::id();
        let device = world.getattr("device")?.str()?.to_str()?.to_owned();
        let fields = crate::cnot_acceptance::contract_fields(&contract)?;
        crate::cnot_acceptance::emit(
            "backend_admission",
            rank,
            size,
            pid,
            &device,
            &fields,
            gpu_observation.as_ref(),
        )?;
        let controller_fields = [
            ("implementation".to_owned(), "RustNativeStages".to_owned()),
            ("abi".to_owned(), "1".to_owned()),
            ("active".to_owned(), "1".to_owned()),
        ]
        .into_iter()
        .collect();
        crate::cnot_acceptance::emit(
            "rust_controller",
            rank,
            size,
            pid,
            &device,
            &controller_fields,
            gpu_observation.as_ref(),
        )?;
        if std::env::var_os("MGBS_CNOT_ACCEPTANCE_RUN_ID").is_some() && size == 1 {
            let single_rank_fields = [
                ("reason".to_owned(), "single_rank".to_owned()),
                ("exchange".to_owned(), "not_applicable".to_owned()),
            ]
            .into_iter()
            .collect();
            crate::cnot_acceptance::emit(
                "stream5_absent",
                rank,
                size,
                pid,
                &device,
                &single_rank_fields,
                gpu_observation.as_ref(),
            )?;
        }
        Ok(py.None().into_bound(py))
    })();
    checked(py, world, "write CNOT admission receipts", receipt_write)?;
    Ok(admitted)
}

pub(crate) fn bind_stream5_admission<'py>(
    py: Python<'py>,
    admission: &Py<PyCnotBackendAdmission>,
    world: &Bound<'py, PyAny>,
    library_sha256: Option<String>,
) -> PyResult<()> {
    if !admission.borrow(py).strict {
        return Ok(());
    }
    let size: usize = world.getattr("size")?.extract()?;
    if size <= 1 {
        return Ok(());
    }
    let digest = library_sha256.ok_or_else(|| {
        PyRuntimeError::new_err("strict CNOT admission lacks the Rust Stream 5 fingerprint")
    })?;
    let transport = world.getattr("transport")?;
    if transport.is_none() {
        return Err(PyRuntimeError::new_err(
            "strict CNOT admission lacks its live Rust Stream 5 transport",
        ));
    }
    let mut admission = admission.borrow_mut(py);
    admission.stream5_library_sha256 = Some(digest);
    admission.stream5_transport = Some(transport.unbind());
    Ok(())
}

fn required<'py>(value: &Option<Bound<'py, PyAny>>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    value
        .clone()
        .ok_or_else(|| PyRuntimeError::new_err(format!("CNOT stage {name} is unavailable")))
}

fn slice<'py>(
    py: Python<'py>,
    begin: Option<i64>,
    end: Option<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("builtins")?.getattr("slice")?.call1((begin, end))
}

fn column<'py>(tensor: &Bound<'py, PyAny>, index: i64) -> PyResult<Bound<'py, PyAny>> {
    tensor.get_item((slice(tensor.py(), None, None)?, index))
}

struct NativeStages<'py> {
    py: Python<'py>,
    world: Bound<'py, PyAny>,
    root: Bound<'py, PyAny>,
    identity: Bound<'py, PyAny>,
    input: Bound<'py, PyAny>,
    scorer: Bound<'py, PyAny>,
    mode: String,
    audit: bool,
    n: i64,
    words: i64,
    rank: u32,
    size: u32,
    parents: Option<Bound<'py, PyAny>>,
    lex: Option<Bound<'py, PyAny>>,
    closed: Option<Bound<'py, PyAny>>,
    candidates: Option<Bound<'py, PyAny>>,
    owners: Option<Bound<'py, PyAny>>,
    incoming: Option<Bound<'py, PyAny>>,
    children: Option<Bound<'py, PyAny>>,
    meta: Option<Bound<'py, PyAny>>,
    values: Option<Bound<'py, PyAny>>,
    ranks: Option<Bound<'py, PyAny>>,
    local: Option<Bound<'py, PyAny>>,
    new_lex: Option<Bound<'py, PyAny>>,
    device_domain_path: Option<PathBuf>,
    strict_backend: bool,
    device_domain_sha256: Option<String>,
    gpu_observation: Option<crate::cnot_acceptance::GpuObservation>,
    stream5_library_sha256: Option<String>,
    device_domain: Option<beam_native_sys::cnot_device_bridge::DeviceDomain>,
    _stream5_transport: Option<Py<PyAny>>,
}

impl<'py> NativeStages<'py> {
    fn new(
        py: Python<'py>,
        world: Bound<'py, PyAny>,
        root: Bound<'py, PyAny>,
        identity: Bound<'py, PyAny>,
        input: Bound<'py, PyAny>,
        scorer: Bound<'py, PyAny>,
        mode: String,
        audit: bool,
        admission: Option<Py<PyCnotBackendAdmission>>,
        admitted_topology: Option<(u32, u32)>,
        ranked_request: Option<RankedCnotRequest>,
    ) -> PyResult<Self> {
        let python_rank: u32 = world.getattr("rank")?.extract()?;
        let python_size: u32 = world.getattr("size")?.extract()?;
        let (rank, size) = if let Some(request) = ranked_request {
            let topology = (request.rank(), request.world());
            if topology != (python_rank, python_size)
                || admitted_topology.is_some_and(|value| value != topology)
            {
                return Err(PyRuntimeError::new_err(format!(
                    "ranked CNOT request topology (rank={}, world={}) does not match Python World (rank={python_rank}, world={python_size})",
                    topology.0, topology.1
                )));
            }
            (topology.0, topology.1)
        } else {
            match admitted_topology {
                None => (python_rank, python_size),
                Some((rank, size)) if rank == python_rank && size == python_size => (rank, size),
                Some((rank, size)) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "admitted CNOT rank topology (rank={rank}, world={size}) does not match Python World (rank={python_rank}, world={python_size})"
                    )));
                }
            }
        };
        let (mode, audit) = match ranked_request {
            Some(request) => (request.mode().as_str().to_owned(), request.audit()),
            None => (mode, audit),
        };
        if size == 0 || rank >= size {
            return Err(PyValueError::new_err("invalid admitted CNOT rank topology"));
        }
        let n = root.call_method1("size", (1,))?.extract()?;
        let words = root.call_method1("size", (2,))?.extract()?;
        let (
            strict_backend,
            device_domain,
            device_domain_sha256,
            gpu_observation,
            stream5_library_sha256,
            stream5_transport,
        ) = if let Some(admission) = admission {
            let mut admission = admission.borrow_mut(py);
            (
                admission.strict,
                admission.device_domain.take(),
                admission.device_domain_sha256.take(),
                admission.gpu_observation.take(),
                admission.stream5_library_sha256.take(),
                admission.stream5_transport.take(),
            )
        } else {
            (strict_backend_enabled(), None, None, None, None, None)
        };
        if strict_backend && device_domain.is_none() {
            return Err(PyRuntimeError::new_err(
                "strict CNOT acceptance requires its admitted DeviceDomain handle",
            ));
        }
        if strict_backend
            && size > 1
            && (stream5_library_sha256.is_none() || stream5_transport.is_none())
        {
            return Err(PyRuntimeError::new_err(
                "strict CNOT acceptance requires its admitted Stream 5 handle and fingerprint",
            ));
        }
        let device_domain_path = if strict_backend {
            None
        } else {
            std::env::var_os("MGBS_CNOT_OWNED_LIBRARY").map(PathBuf::from)
        };
        Ok(Self {
            py,
            world,
            root,
            identity,
            input,
            scorer,
            mode,
            audit,
            n,
            words,
            rank,
            size,
            parents: None,
            lex: None,
            closed: None,
            candidates: None,
            owners: None,
            incoming: None,
            children: None,
            meta: None,
            values: None,
            ranks: None,
            local: None,
            new_lex: None,
            device_domain_path,
            strict_backend,
            device_domain_sha256,
            gpu_observation,
            stream5_library_sha256,
            device_domain,
            _stream5_transport: stream5_transport,
        })
    }

    fn native(&self) -> PyResult<Bound<'py, PyModule>> {
        self.py.import("multigpubeamsearch.cnot_native")
    }

    fn ops(&self, tensor: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if tensor.getattr("is_cuda")?.extract()? {
            Ok(Some(self.native()?.call_method0("_cuda_ops")?))
        } else {
            Ok(None)
        }
    }

    fn validate_current_gpu_observation(&self) -> PyResult<()> {
        let device = self.world.getattr("device")?;
        let observed = observe_acceptance_gpu(self.py, &self.world, &device)?;
        if observed != self.gpu_observation {
            return Err(PyRuntimeError::new_err(
                "CUDA context changed after CNOT backend admission",
            ));
        }
        Ok(())
    }

    fn ensure_device_domain(&mut self, tensor: &Bound<'py, PyAny>) -> PyResult<bool> {
        if self.strict_backend {
            self.validate_current_gpu_observation()?;
        }
        if !tensor.getattr("is_cuda")?.extract::<bool>()? {
            if self.strict_backend {
                return Err(PyRuntimeError::new_err(
                    "strict CNOT acceptance cannot fall back from CUDA DeviceDomain",
                ));
            }
            return Ok(false);
        }
        if self.device_domain.is_some() {
            return Ok(true);
        }
        if self.strict_backend {
            return Err(PyRuntimeError::new_err(
                "strict CNOT acceptance cannot load an unadmitted DeviceDomain",
            ));
        }
        let Some(path) = &self.device_domain_path else {
            return Ok(false);
        };
        let device = tensor.getattr("device")?.getattr("index")?.extract()?;
        self.device_domain = Some(beam_native_sys::cnot_device_bridge::DeviceDomain::load(
            path,
            device,
            self.n as i32,
            self.size as i32,
        )?);
        Ok(true)
    }

    fn trace_device_dispatch(&self, operation: &str, rows: usize) -> PyResult<()> {
        self.validate_current_gpu_observation()?;
        emit_device_dispatch(
            &self.world,
            self.rank,
            self.size,
            self.strict_backend,
            self.device_domain_sha256.as_deref(),
            operation,
            rows,
            self.gpu_observation.as_ref(),
        )
    }

    fn expand_batch(&mut self, begin: u64, k: u32) -> PyResult<Bound<'py, PyAny>> {
        let parents = required(&self.parents, "parents")?;
        let lex = required(&self.lex, "lex")?;
        if self.ensure_device_domain(&parents)? {
            let parent_count = parents.len()?;
            let expanded = cnot_native_batch::expand(
                self.py,
                &self.world,
                &parents,
                &lex,
                begin,
                &self.scorer,
                &self.mode,
                self.strict_backend,
                k,
                self.n,
                self.words,
                self.device_domain.as_mut(),
            )?;
            let rows = parent_count
                .checked_mul(k as usize)
                .ok_or_else(|| PyRuntimeError::new_err("CNOT expansion row count overflow"))?;
            self.trace_device_dispatch("expansion", rows)?;
            return Ok(expanded);
        }
        let ops = self.ops(&parents)?;
        let native = match &ops {
            Some(ops) => ops.hasattr("expand_candidates")?,
            None => false,
        };
        if native {
            let ops =
                ops.ok_or_else(|| PyRuntimeError::new_err("CNOT expansion operators unavailable"))?;
            let count = parents.len()?;
            let alphabet = self.n * (self.n - 1);
            let scores = if count > 0 && i64::from(k) < alphabet {
                let raw = cnot_scorer_dispatch::parent_scores(
                    self.py,
                    &self.world,
                    &self.scorer,
                    &parents,
                    &self.mode,
                    self.strict_backend,
                )?;
                cnot_native_batch::scores(
                    self.py,
                    raw,
                    &[count as i64, alphabet],
                    &self.world.getattr("device")?,
                )?
            } else {
                self.py.None().into_bound(self.py)
            };
            ops.call_method1(
                "expand_candidates",
                (
                    parents,
                    lex,
                    scores,
                    begin,
                    self.rank,
                    self.size,
                    k,
                    self.mode == "proposal",
                ),
            )
        } else {
            cnot_native_batch::expand(
                self.py,
                &self.world,
                &parents,
                &lex,
                begin,
                &self.scorer,
                &self.mode,
                self.strict_backend,
                k,
                self.n,
                self.words,
                None,
            )
        }
    }

    fn distinct_batch(&mut self) -> PyResult<Bound<'py, PyAny>> {
        let incoming = required(&self.incoming, "incoming")?;
        let closed = required(&self.closed, "closed")?;
        if self.ensure_device_domain(&incoming)? {
            let input_rows = incoming.len()?;
            let distinct = cnot_native_batch::distinct(
                self.py,
                &self.world,
                &incoming,
                &closed,
                self.n,
                self.words,
                &self.scorer,
                self.strict_backend,
                self.device_domain.as_mut(),
            )?;
            self.trace_device_dispatch("full_matrix_distinct", input_rows)?;
            return Ok(distinct);
        }
        let ops = self.ops(&incoming)?;
        let native = match &ops {
            Some(ops) => ops.hasattr("distinct_candidates")?,
            None => false,
        };
        if native {
            let ops =
                ops.ok_or_else(|| PyRuntimeError::new_err("CNOT distinct operators unavailable"))?;
            let pair = ops.call_method1(
                "distinct_candidates",
                (&incoming, &closed, self.n, self.words),
            )?;
            let children = pair.get_item(0)?;
            let meta = pair.get_item(1)?;
            let values = if children.len()? > 0 {
                let raw = cnot_scorer_dispatch::child_values(
                    self.py,
                    &self.world,
                    &self.scorer,
                    &children,
                    self.strict_backend,
                )?;
                cnot_native_batch::scores(
                    self.py,
                    raw,
                    &[children.len()? as i64],
                    &self.world.getattr("device")?,
                )?
            } else {
                let torch = self.py.import("torch")?;
                let kwargs = PyDict::new(self.py);
                kwargs.set_item("dtype", torch.getattr("float64")?)?;
                kwargs.set_item("device", self.world.getattr("device")?)?;
                torch.call_method("empty", (0,), Some(&kwargs))?
            };
            Ok(PyTuple::new(self.py, [children, meta, values])?.into_any())
        } else {
            cnot_native_batch::distinct(
                self.py,
                &self.world,
                &incoming,
                &closed,
                self.n,
                self.words,
                &self.scorer,
                self.strict_backend,
                None,
            )
        }
    }

    fn exchange(
        &self,
        rows: &Bound<'py, PyAny>,
        targets: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.strict_backend && self.size > 1 {
            let name: String = self
                .world
                .getattr("transport")?
                .getattr("__class__")?
                .getattr("__name__")?
                .extract()?;
            if name != "RustStream5" {
                return Err(PyRuntimeError::new_err(
                    "strict CNOT acceptance cannot dispatch through a non-Rust Stream 5 transport",
                ));
            }
        }
        let received = cnot_native_exchange::exchange(self.py, &self.world, rows, targets, true)?;
        if self.strict_backend && self.size > 1 {
            let device = self.world.getattr("device")?.str()?.to_str()?.to_owned();
            let sha = self.stream5_library_sha256.as_deref().unwrap_or("unknown");
            let fields = [
                ("implementation".to_owned(), "RustStream5/NCCL".to_owned()),
                ("operation".to_owned(), "exchange".to_owned()),
                ("abi".to_owned(), "1".to_owned()),
                ("owned_abi".to_owned(), "1".to_owned()),
                ("exchange_abi".to_owned(), "1".to_owned()),
                ("library_sha256".to_owned(), sha.to_owned()),
                ("rows".to_owned(), rows.len()?.to_string()),
            ]
            .into_iter()
            .collect();
            let receipt = crate::cnot_acceptance::emit(
                "stream5_dispatch",
                self.rank,
                self.size,
                std::process::id(),
                &device,
                &fields,
                self.gpu_observation.as_ref(),
            )
            .map(|()| self.py.None().into_bound(self.py));
            checked(
                self.py,
                &self.world,
                "write Rust Stream 5 exchange receipt",
                receipt,
            )?;
        }
        Ok(received)
    }

    fn fallback_winner_packet(
        &self,
        ranks: &Bound<'py, PyAny>,
        local: &Bound<'py, PyAny>,
        new_lex: &Bound<'py, PyAny>,
        meta: &Bound<'py, PyAny>,
        children: &Bound<'py, PyAny>,
        closed: &Bound<'py, PyAny>,
        count: usize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let torch = self.py.import("torch")?;
        let slots = ranks
            .call_method1("__eq__", (self.rank,))?
            .call_method0("nonzero")?
            .call_method0("flatten")?;
        let kept = local.get_item(&slots)?;
        let owned = children.get_item(&kept)?;
        let new_closed = torch.call_method1("cat", (PyTuple::new(self.py, [closed, &owned])?,))?;
        let columns = PyTuple::new(
            self.py,
            [
                slots.clone(),
                new_lex.get_item(&slots)?,
                meta.get_item((&kept, 0))?,
                meta.get_item((&kept, 3))?,
                ranks.get_item(&slots)?,
            ],
        )?;
        let records = torch.call_method1("stack", (columns, 1))?;
        let rows = owned.call_method1("reshape", (kept.len()?, self.n * self.words))?;
        let wire = torch.call_method1("cat", (PyTuple::new(self.py, [records, rows])?, 1))?;
        let targets = slots
            .call_method1("__mul__", (self.size,))?
            .call_method1("__floordiv__", (count.max(1),))?;
        Ok((new_closed, wire, targets))
    }
}

#[pyfunction(name = "_probe_cnot_device_stage_vote")]
fn probe_cnot_device_stage_vote(world: Bound<'_, PyAny>, fail_rank: i64) -> PyResult<()> {
    let py = world.py();
    let rank: u32 = world.getattr("rank")?.extract()?;
    let size: u32 = world.getattr("size")?.extract()?;
    let local = (|| -> PyResult<Bound<'_, PyAny>> {
        if i64::from(rank) == fail_rank {
            return Err(PyRuntimeError::new_err(
                "injected DeviceDomain stage failure before dispatch trace",
            ));
        }
        // Use the exact local-only event helper called by the production
        // owners/expand/distinct stages. The enclosing vote is their sole
        // collective for this rank-local stage.
        emit_device_dispatch(
            &world,
            rank,
            size,
            true,
            Some("fixture"),
            "expansion",
            1,
            None,
        )?;
        Ok(py.None().into_bound(py))
    })();
    checked(py, &world, "rank parent actions and expand", local)?;
    Ok(())
}

#[pyfunction(name = "_probe_cnot_device_domain_vote")]
fn probe_cnot_device_domain_vote(
    py: Python<'_>,
    world: Bound<'_, PyAny>,
    path: String,
    device: i32,
    n: i32,
    fail_rank: i64,
) -> PyResult<String> {
    let world_size: i32 = world.getattr("size")?.extract()?;
    let path = PathBuf::from(path);
    let (_domain, domain_sha256) = prepare_device_domain_checked(
        py,
        &world,
        || load_device_domain_admission(&path, device, n, world_size),
        (fail_rank >= 0).then_some(fail_rank),
    )?;
    Ok(format!("domain_abi=1;domain_sha256={domain_sha256}"))
}

impl Backend for NativeStages<'_> {
    type Error = PyErr;
    type Layer = Py<PyAny>;
    type Records = Py<PyAny>;

    fn root_is_goal(&mut self) -> PyResult<bool> {
        let goal = (|| -> PyResult<Bound<'_, PyAny>> {
            if let Some(is_identity) = crate::cnot::search_input_is_identity(&self.input)? {
                return self
                    .py
                    .import("builtins")?
                    .getattr("bool")?
                    .call1((is_identity,));
            }
            self.py
                .import("torch")?
                .call_method1("equal", (&self.root, &self.identity))
        })();
        checked(self.py, &self.world, "check CNOT root goal", goal)?.extract()
    }

    fn initialize(&mut self) -> PyResult<()> {
        let prepared = checked(
            self.py,
            &self.world,
            "prepare initial CNOT frontier",
            (|| -> PyResult<Bound<'_, PyAny>> {
                let empty = self.root.call_method1("narrow", (0, 0, 0))?;
                let parents = if self.rank == 0 {
                    self.root.clone()
                } else {
                    empty.clone()
                };
                let initial: Vec<i64> = if self.rank == 0 { vec![0] } else { Vec::new() };
                let lex = self.world.call_method1("longs", (initial,))?;
                Ok(PyTuple::new(self.py, [parents, lex, empty])?.into_any())
            })(),
        )?;
        self.parents = Some(prepared.get_item(0)?);
        self.lex = Some(prepared.get_item(1)?);
        let empty = prepared.get_item(2)?;
        let root = self.root.clone();
        let owner = (|| {
            if self.ensure_device_domain(&root)? {
                let domain = self
                    .device_domain
                    .as_mut()
                    .ok_or_else(|| PyRuntimeError::new_err("CNOT device domain is unavailable"))?;
                let owners = domain.owners(self.py, &root)?;
                self.trace_device_dispatch("owners", 1)?;
                owners.call_method0("item")
            } else if root.getattr("is_cuda")?.is_truthy()? {
                self.native()?
                    .call_method1("cnot_owners", (&root, self.size))?
                    .call_method0("item")
            } else {
                crate::cnot_native_batch::cpu_owners(self.py, &root, self.size)?
                    .call_method0("item")
            }
        })();
        let owner: u32 =
            checked(self.py, &self.world, "initialize CNOT domain", owner)?.extract()?;
        self.closed = Some(if owner == self.rank {
            self.root.clone()
        } else {
            empty
        });
        Ok(())
    }

    fn goal(&mut self, begin: u64) -> PyResult<Option<Goal>> {
        let parents = required(&self.parents, "parents")?;
        let lex = required(&self.lex, "lex")?;
        let goal = crate::cnot_native_goal::goal(
            self.py,
            &self.world,
            &parents,
            &lex,
            begin,
            &self.identity,
        )?;
        Ok(goal.map(|(slot, movement)| Goal { slot, movement }))
    }

    fn expand(&mut self, begin: u64, k: u32) -> PyResult<()> {
        let batch = self.expand_batch(begin, k);
        let pair = checked(
            self.py,
            &self.world,
            "rank parent actions and expand",
            batch,
        )?;
        self.candidates = Some(pair.get_item(0)?);
        self.owners = Some(pair.get_item(1)?);
        Ok(())
    }

    fn exchange_candidates(&mut self) -> PyResult<()> {
        self.incoming = Some(self.exchange(
            &required(&self.candidates, "candidates")?,
            &required(&self.owners, "owners")?,
        )?);
        Ok(())
    }

    fn deduplicate_and_score(&mut self) -> PyResult<()> {
        let batch = self.distinct_batch();
        let result = checked(
            self.py,
            &self.world,
            "deduplicate and evaluate distinct children",
            batch,
        )?;
        self.children = Some(result.get_item(0)?);
        self.meta = Some(result.get_item(1)?);
        self.values = Some(result.get_item(2)?);
        Ok(())
    }

    fn select(&mut self, width: u64) -> PyResult<()> {
        let inputs = checked(
            self.py,
            &self.world,
            "prepare CNOT selection inputs",
            (|| -> PyResult<Bound<'_, PyAny>> {
                let values = required(&self.values, "values")?;
                let word_keys = column(&required(&self.meta, "meta")?, 1)?;
                Ok(PyTuple::new(self.py, [values, word_keys])?.into_any())
            })(),
        )?;
        let values = inputs.get_item(0)?;
        let word_keys = inputs.get_item(1)?;

        let result = crate::cnot_native_selection::select(
            self.py,
            self.world.clone(),
            values,
            word_keys,
            width,
        )?
        .into_bound(self.py);
        let output = checked(
            self.py,
            &self.world,
            "prepare CNOT selected frontier",
            (|| -> PyResult<Bound<'_, PyAny>> {
                Ok(PyTuple::new(
                    self.py,
                    [
                        result.get_item(0)?,
                        result.get_item(1)?,
                        result.get_item(2)?,
                    ],
                )?
                .into_any())
            })(),
        )?;
        self.ranks = Some(output.get_item(0)?);
        self.local = Some(output.get_item(1)?);
        self.new_lex = Some(output.get_item(2)?);
        Ok(())
    }

    fn redistribute(&mut self, depth: u64) -> PyResult<Round<Self::Layer, Self::Records>> {
        let packet = checked(
            self.py,
            &self.world,
            "prepare CNOT winner packet",
            (|| -> PyResult<Bound<'_, PyAny>> {
                let ranks = required(&self.ranks, "ranks")?;
                let local = required(&self.local, "local")?;
                let new_lex = required(&self.new_lex, "new_lex")?;
                let meta = required(&self.meta, "meta")?;
                let children = required(&self.children, "children")?;
                let closed = required(&self.closed, "closed")?;
                let count = local.len()?;
                let ops = self.ops(&children)?;
                let use_native = match &ops {
                    Some(ops) => {
                        ops.hasattr("winner_packet")?
                            && ops.hasattr("sort_received")?
                            && ops.hasattr("frontier_views")?
                    }
                    None => false,
                };
                let (new_closed, wire, targets) = if use_native {
                    let packet = ops
                        .as_ref()
                        .ok_or_else(|| {
                            PyRuntimeError::new_err("CNOT winner operators unavailable")
                        })?
                        .call_method1(
                            "winner_packet",
                            (
                                &ranks, &local, &new_lex, &meta, &children, &closed, self.rank,
                                self.size,
                            ),
                        )?;
                    (
                        packet.get_item(0)?,
                        packet.get_item(1)?,
                        packet.get_item(2)?,
                    )
                } else {
                    self.fallback_winner_packet(
                        &ranks, &local, &new_lex, &meta, &children, &closed, count,
                    )?
                };
                let builtins = self.py.import("builtins")?;
                let count = builtins.getattr("int")?.call1((count,))?;
                let native = builtins.getattr("bool")?.call1((use_native,))?;
                let ops = ops.unwrap_or_else(|| self.py.None().into_bound(self.py));
                Ok(
                    PyTuple::new(self.py, [new_closed, wire, targets, ops, native, count])?
                        .into_any(),
                )
            })(),
        )?;
        let new_closed = packet.get_item(0)?;
        let wire = packet.get_item(1)?;
        let targets = packet.get_item(2)?;
        let ops_value = packet.get_item(3)?;
        let ops = (!ops_value.is_none()).then_some(ops_value);
        let use_native: bool = packet.get_item(4)?.extract()?;
        let count: usize = packet.get_item(5)?.extract()?;
        let received = self.exchange(&wire, &targets)?;
        let received = checked(
            self.py,
            &self.world,
            "prepare sorted CNOT winner packet",
            (|| -> PyResult<Bound<'_, PyAny>> {
                if use_native {
                    ops.as_ref()
                        .ok_or_else(|| {
                            PyRuntimeError::new_err("CNOT winner operators unavailable")
                        })?
                        .call_method1("sort_received", (&received,))
                } else {
                    let order = self
                        .py
                        .import("torch")?
                        .call_method1("argsort", (column(&received, 0)?,))?;
                    received.get_item(order)
                }
            })(),
        )?;
        let local_stats = checked(
            self.py,
            &self.world,
            "prepare CNOT redistribution statistics",
            (|| -> PyResult<Bound<'_, PyAny>> {
                let parents = required(&self.parents, "parents")?;
                let candidates = required(&self.candidates, "candidates")?;
                let owners = required(&self.owners, "owners")?;
                let children = required(&self.children, "children")?;
                let remote: i64 = owners
                    .call_method1("__ne__", (self.rank,))?
                    .call_method0("sum")?
                    .call_method0("item")?
                    .extract()?;
                let stat_values = vec![
                    i64::try_from(parents.len()?)
                        .map_err(|_| PyOverflowError::new_err("CNOT parent count exceeds int64"))?,
                    i64::try_from(candidates.len()?).map_err(|_| {
                        PyOverflowError::new_err("CNOT candidate count exceeds int64")
                    })?,
                    remote,
                    i64::try_from(children.len()?).map_err(|_| {
                        PyOverflowError::new_err("CNOT distinct count exceeds int64")
                    })?,
                    i64::try_from(received.len()?)
                        .map_err(|_| PyOverflowError::new_err("CNOT winner count exceeds int64"))?,
                ];
                self.world.call_method1("longs", (stat_values,))
            })(),
        )?;
        let gathered_stats = self.world.call_method1("gather_fixed", (&local_stats,));
        let gathered_stats = checked(
            self.py,
            &self.world,
            "gather CNOT redistribution statistics",
            gathered_stats,
        )?;
        let summary = checked(
            self.py,
            &self.world,
            "materialize CNOT redistribution statistics",
            (|| -> PyResult<Bound<'_, PyAny>> {
                let stats = gathered_stats.call_method0("tolist")?;
                let rows: Vec<Vec<i64>> = stats.extract()?;
                let distinct = rows.iter().try_fold(0_i64, |sum, row| {
                    let count = row
                        .get(3)
                        .ok_or_else(|| PyRuntimeError::new_err("invalid CNOT rank statistics"))?;
                    sum.checked_add(*count)
                        .ok_or_else(|| PyOverflowError::new_err("CNOT rank statistics overflow"))
                })?;
                let distinct = self
                    .py
                    .import("builtins")?
                    .getattr("int")?
                    .call1((distinct,))?;
                Ok(PyTuple::new(self.py, [distinct, stats])?.into_any())
            })(),
        )?;
        let distinct = summary.get_item(0)?;
        let stats = summary.get_item(1)?;
        let layer = checked(
            self.py,
            &self.world,
            "prepare CNOT layer report",
            (|| -> PyResult<Bound<'_, PyAny>> {
                let layer = PyDict::new(self.py);
                layer.set_item("depth", depth)?;
                layer.set_item("by_rank", &stats)?;
                layer.set_item("distinct", &distinct)?;
                layer.set_item("retained", count)?;
                Ok(layer.into_any())
            })(),
        )?;
        if self.audit {
            let payload = checked(
                self.py,
                &self.world,
                "prepare CNOT audit payload",
                (|| -> PyResult<Bound<'_, PyAny>> {
                    let torch = self.py.import("torch")?;
                    let values = required(&self.values, "values")?;
                    let meta = required(&self.meta, "meta")?;
                    let children = required(&self.children, "children")?;
                    let values = values
                        .call_method1("to", (torch.getattr("float64")?,))?
                        .call_method0("contiguous")?
                        .call_method1("view", (torch.getattr("int64")?,))?
                        .call_method1("unsqueeze", (1,))?;
                    let flat =
                        children.call_method1("reshape", (children.len()?, self.n * self.words))?;
                    Ok(torch
                        .call_method1("cat", (PyTuple::new(self.py, [values, meta, flat])?, 1))?
                        .into_any())
                })(),
            )?;
            let candidate_rows = self.world.call_method1("gather_rows", (&payload,));
            let candidate_rows = checked(
                self.py,
                &self.world,
                "gather CNOT candidate audit rows",
                candidate_rows,
            )?;
            let candidates = checked(
                self.py,
                &self.world,
                "materialize CNOT candidate audit rows",
                (|| -> PyResult<Bound<'_, PyAny>> {
                    candidate_rows.call_method0("cpu")?.call_method0("tolist")
                })(),
            )?;
            checked(
                self.py,
                &self.world,
                "store CNOT candidate audit rows",
                (|| -> PyResult<Bound<'_, PyAny>> {
                    layer.set_item("candidates", candidates)?;
                    Ok(self.py.None().into_bound(self.py))
                })(),
            )?;
            let winner_rows = self.world.call_method1("gather_rows", (&received,));
            let winner_rows = checked(
                self.py,
                &self.world,
                "gather CNOT winner audit rows",
                winner_rows,
            )?;
            let winners = checked(
                self.py,
                &self.world,
                "materialize CNOT winner audit rows",
                (|| -> PyResult<Bound<'_, PyAny>> {
                    winner_rows.call_method0("cpu")?.call_method0("tolist")
                })(),
            )?;
            checked(
                self.py,
                &self.world,
                "store CNOT winner audit rows",
                (|| -> PyResult<Bound<'_, PyAny>> {
                    layer.set_item("winners", winners)?;
                    Ok(self.py.None().into_bound(self.py))
                })(),
            )?;
        }
        if count == 0 {
            self.closed = Some(new_closed);
            return Ok(Round {
                count: 0,
                layer: layer.into_any().unbind(),
                records: None,
            });
        }
        let frontier = checked(
            self.py,
            &self.world,
            "prepare CNOT next frontier and history rows",
            (|| -> PyResult<Bound<'_, PyAny>> {
                if use_native {
                    let parts = ops
                        .as_ref()
                        .ok_or_else(|| {
                            PyRuntimeError::new_err("CNOT winner operators unavailable")
                        })?
                        .call_method1("frontier_views", (&received, self.n, self.words))?;
                    Ok(PyTuple::new(
                        self.py,
                        [parts.get_item(0)?, parts.get_item(1)?, parts.get_item(2)?],
                    )?
                    .into_any())
                } else {
                    let history = received
                        .get_item((
                            slice(self.py, None, None)?,
                            slice(self.py, Some(2), Some(5))?,
                        ))?
                        .call_method0("contiguous")?;
                    let next = received
                        .get_item((slice(self.py, None, None)?, slice(self.py, Some(5), None)?))?
                        .call_method1("reshape", (-1, self.n, self.words))?
                        .call_method0("contiguous")?;
                    let lex = column(&received, 1)?.call_method0("contiguous")?;
                    Ok(PyTuple::new(self.py, [history, next, lex])?.into_any())
                }
            })(),
        )?;
        let history = frontier.get_item(0)?;
        let next_parents = frontier.get_item(1)?;
        let lex = frontier.get_item(2)?;
        self.closed = Some(new_closed);
        self.parents = Some(next_parents);
        self.lex = Some(lex);
        Ok(Round {
            count: count as u64,
            layer: layer.into_any().unbind(),
            records: Some(history.unbind()),
        })
    }

    fn history_edge(
        &mut self,
        records: &Self::Records,
        target: u32,
        local_index: u64,
    ) -> PyResult<Edge> {
        let edge = checked(
            self.py,
            &self.world,
            "prepare CNOT history edge",
            (|| -> PyResult<Bound<'_, PyAny>> {
                let edge = self.world.call_method1("longs", (vec![0_i64, 0, 0],))?;
                if target == self.rank {
                    let record = records.bind(self.py).get_item(local_index)?;
                    edge.call_method1("copy_", (record,))?;
                }
                Ok(edge)
            })(),
        )?;
        let broadcast = self.world.call_method1("broadcast", (&edge, target));
        let edge = checked(
            self.py,
            &self.world,
            "broadcast CNOT history edge",
            broadcast,
        )?;
        let parsed = checked(
            self.py,
            &self.world,
            "validate CNOT history edge",
            (|| -> PyResult<Bound<'_, PyAny>> {
                let row: Vec<i64> = edge.call_method0("tolist")?.extract()?;
                if row.len() != 3 {
                    return Err(PyValueError::new_err("invalid CNOT history edge"));
                }
                let parent = u64::try_from(row[0])
                    .map_err(|_| PyValueError::new_err("invalid CNOT history parent"))?;
                let route = u32::try_from(row[1])
                    .map_err(|_| PyValueError::new_err("invalid CNOT history route"))?;
                let owner = u32::try_from(row[2])
                    .map_err(|_| PyValueError::new_err("invalid CNOT history owner"))?;
                let int = self.py.import("builtins")?.getattr("int")?;
                let parent = int.call1((parent,))?;
                let route = int.call1((route,))?;
                let owner = int.call1((owner,))?;
                Ok(PyTuple::new(self.py, [parent, route, owner])?.into_any())
            })(),
        )?;
        Ok(Edge {
            parent: parsed.get_item(0)?.extract()?,
            route: parsed.get_item(1)?.extract()?,
            owner: parsed.get_item(2)?.extract()?,
        })
    }

    fn verifies_construction(&mut self, gates: &[u32]) -> PyResult<bool> {
        if let Some(valid) =
            crate::cnot::search_input_verifies_construction(self.py, &self.input, gates)?
        {
            return Ok(valid);
        }
        self.input
            .call_method1("verifies_construction", (gates.to_vec(),))?
            .extract()
    }

    fn invalid(&self, message: &'static str) -> PyErr {
        PyRuntimeError::new_err(message)
    }
}

#[pyfunction]
#[pyo3(signature=(world, root, identity, input, scorer, mode, audit, width, shortlist, alphabet, max_depth, admission=None, admitted_topology=None, ranked_request=None))]
#[allow(clippy::too_many_arguments)]
fn run_cnot_native<'py>(
    py: Python<'py>,
    world: Bound<'py, PyAny>,
    root: Bound<'py, PyAny>,
    identity: Bound<'py, PyAny>,
    input: Bound<'py, PyAny>,
    scorer: Bound<'py, PyAny>,
    mode: String,
    audit: bool,
    width: u64,
    shortlist: u32,
    alphabet: u32,
    max_depth: u64,
    admission: Option<Py<PyCnotBackendAdmission>>,
    admitted_topology: Option<(u32, u32)>,
    ranked_request: Option<Py<PyCnotRankedRequest>>,
) -> PyResult<Py<PyAny>> {
    let typed_request = ranked_request
        .as_ref()
        .map(|request| request.borrow(py).inner);
    let mut backend = None;
    let prepare_backend = (|| -> PyResult<Bound<'py, PyAny>> {
        backend = Some(NativeStages::new(
            py,
            world.clone(),
            root,
            identity,
            input,
            scorer,
            mode,
            audit,
            admission,
            admitted_topology,
            typed_request,
        )?);
        Ok(py.None().into_bound(py))
    })();
    checked(
        py,
        &world,
        "bind admitted CNOT rank topology",
        prepare_backend,
    )?;
    let mut backend = backend.ok_or_else(|| {
        PyRuntimeError::new_err("CNOT rank-topology vote returned no native stages")
    })?;
    let options = typed_request.map_or_else(
        || Options {
            rank: backend.rank,
            world: backend.size,
            width,
            shortlist,
            alphabet,
            max_depth,
        },
        |request| request.search_options(),
    );
    crate::cnot::outcome_to_python(py, cnot_search::search(&mut backend, options)?)
}

#[pyfunction]
pub(crate) fn run_cnot_native_all_ranks_support(
    world: Bound<'_, PyAny>,
    local_support: bool,
) -> PyResult<bool> {
    // Capability negotiation is control traffic, so it uses World’s stable
    // CPU/Gloo group rather than the search transport.
    world
        .call_method1("all_ranks_support", (local_support,))?
        .extract()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCnotBackendAdmission>()?;
    module.add_function(wrap_pyfunction!(run_cnot_native, module)?)?;
    module.add_function(wrap_pyfunction!(probe_cnot_device_stage_vote, module)?)?;
    module.add_function(wrap_pyfunction!(probe_cnot_device_domain_vote, module)?)?;
    module.add_function(wrap_pyfunction!(run_cnot_native_all_ranks_support, module)?)?;
    module.add_function(wrap_pyfunction!(probe_cnot_backend_admission, module)?)?;
    module.add_function(wrap_pyfunction!(require_cnot_backend_consensus, module)?)
}
