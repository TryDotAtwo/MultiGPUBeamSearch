//! Coarse Python adapter for the owned packed CNOT input contract.
use beam_runtime::cnot_search::{self, Backend, CnotRequest, Edge, Goal, Options, RankedCnotRequest, Round};
use beam_runtime::cnot_selection;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict, PyList, PyString, PyTuple},
};
use sha2::{Digest, Sha256};

#[pyclass(name = "CnotMatrix", frozen)]
struct PyCnotMatrix {
    inner: beam_core::cnot::CnotMatrix,
}

#[pyclass(name = "CnotSearchInput", frozen)]
struct PyCnotSearchInput {
    wires: u32,
    raw: Py<PyBytes>,
    root: Py<PyAny>,
    matrix: beam_core::cnot::CnotMatrix,
}

#[pyclass(name = "_CnotRankedRequest", frozen)]
pub(crate) struct PyCnotRankedRequest {
    pub(crate) inner: RankedCnotRequest,
}

pub(crate) fn search_input_is_identity(input: &Bound<'_, PyAny>) -> PyResult<Option<bool>> {
    if !input.is_instance_of::<PyCnotSearchInput>() {
        return Ok(None);
    }
    let typed = input.extract::<PyRef<'_, PyCnotSearchInput>>()?;
    Ok(Some(typed.matrix.is_identity()))
}

pub(crate) fn search_input_verifies_construction(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    moves: &[u32],
) -> PyResult<Option<bool>> {
    if !input.is_instance_of::<PyCnotSearchInput>() {
        return Ok(None);
    }
    let typed = input.extract::<PyRef<'_, PyCnotSearchInput>>()?;
    let matrix = &typed.matrix;
    let moves = moves.to_vec();
    py.allow_threads(|| matrix.verifies_construction(&moves))
        .map(Some)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn validate_cnot_search_controls(
    wires: u32,
    width: &Bound<'_, PyAny>,
    shortlist: &Bound<'_, PyAny>,
    mode: &str,
) -> PyResult<()> {
    let invalid = || PyValueError::new_err("invalid global width, shortlist, or depth");
    let width = width.extract::<u64>().map_err(|_| invalid())?;
    let shortlist = shortlist.extract::<u64>().map_err(|_| invalid())?;
    beam_core::cnot::validate_search_controls(wires, width, shortlist, mode)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn prepare_cnot_input<'py>(
    py: Python<'py>,
    root: &Bound<'py, PyAny>,
    width: &Bound<'py, PyAny>,
    shortlist: &Bound<'py, PyAny>,
    depth: &Bound<'py, PyAny>,
    mode: &Bound<'py, PyAny>,
) -> PyResult<PyCnotSearchInput> {
    let invalid_shape =
        || PyValueError::new_err("expected one packed int64 matrix [1,n,ceil(n/64)]");
    let torch = py.import("torch")?;
    let expected_dtype = torch.getattr("int64")?;
    let dtype_matches = root.getattr("dtype")?.is(&expected_dtype);
    if !dtype_matches {
        return Err(invalid_shape());
    }
    if root.getattr("ndim")?.extract::<usize>()? != 3
        || root.getattr("shape")?.get_item(0)?.extract::<usize>()? != 1
    {
        return Err(invalid_shape());
    }
    let (_, wires, words): (usize, usize, usize) = root.getattr("shape")?.extract()?;
    if !(2..=256).contains(&wires) || words != wires.div_ceil(64) {
        return Err(PyValueError::new_err(
            "CNOT shape requires 2..256 qubits and ceil(n/64) words per row",
        ));
    }

    let invalid_controls = || PyValueError::new_err("invalid global width, shortlist, or depth");
    let builtins = py.import("builtins")?;
    let type_fn = builtins.getattr("type")?;
    let int_type = builtins.getattr("int")?;
    for value in [width, shortlist, depth] {
        if !type_fn.call1((value,))?.is(&int_type) {
            return Err(invalid_controls());
        }
    }
    if depth.call_method1("__lt__", (0,))?.is_truthy()? {
        return Err(invalid_controls());
    }
    let width = width.extract::<u64>().map_err(|_| invalid_controls())?;
    let alphabet = (wires * (wires - 1)) as u64;
    let shortlist = if shortlist.call_method1("__gt__", (alphabet,))?.is_truthy()? {
        alphabet
    } else {
        shortlist.extract::<u64>().map_err(|_| invalid_controls())?
    };
    let mode_text = mode.extract::<String>()?;
    beam_core::cnot::validate_search_controls(wires as u32, width, shortlist, &mode_text)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    // Retain the exact public snapshot semantics: validation, raw hashing and
    // replay all refer to this owned contiguous tensor after the caller returns.
    let clone_kwargs = PyDict::new(py);
    clone_kwargs.set_item("memory_format", torch.getattr("contiguous_format")?)?;
    let snapshot = root
        .call_method0("detach")?
        .call_method("clone", (), Some(&clone_kwargs))?;
    let raw_object = snapshot
        .call_method0("cpu")?
        .call_method0("numpy")?
        .call_method0("tobytes")?;
    let raw = raw_object.downcast::<PyBytes>()?.as_bytes();
    let inner = beam_core::cnot::CnotMatrix::from_le_bytes(wires as u32, raw)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyCnotSearchInput {
        wires: wires as u32,
        raw: PyBytes::new(py, raw).unbind(),
        root: snapshot.unbind(),
        matrix: inner,
    })
}

#[pyfunction(name = "_cnot_public_dispatch_digest")]
fn cnot_public_dispatch_digest<'py>(
    py: Python<'py>,
    engine: Bound<'py, PyAny>,
    max_depth: Bound<'py, PyAny>,
    strict_backend: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let builtins = py.import("builtins")?;
    let engine_type = builtins.getattr("type")?.call1((&engine,))?;
    if !engine_type.is(&builtins.getattr("str")?) {
        return Err(PyValueError::new_err(
            "CNOT engine must be auto, python, or rust",
        ));
    }
    let engine_text: String = engine.extract()?;
    let depth_fits_u64 = !max_depth.call_method1("__gt__", (u64::MAX,))?.is_truthy()?;
    let digest = beam_runtime::cnot_search::public_dispatch_digest(
        &engine_text,
        depth_fits_u64,
        strict_backend,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &digest))
}

#[pyfunction]
fn run_cnot_native_require_consensus(world: Bound<'_, PyAny>, digest: Vec<u8>) -> PyResult<()> {
    let values: Vec<i64> = digest.into_iter().map(i64::from).collect();
    let local = world.call_method1("longs", (values,))?;
    let gathered = world.call_method1("gather_fixed", (local,))?;
    let first = gathered.get_item(0)?;
    let same = gathered
        .call_method1("__eq__", (first,))?
        .call_method0("all")?
        .call_method0("item")?
        .extract::<bool>()?;
    if !same {
        return Err(PyValueError::new_err(
            "CNOT root, search settings or scorer fingerprints differ between ranks",
        ));
    }
    Ok(())
}

fn selected_engine_name(
    selected: beam_runtime::cnot_search::SelectedEngine,
) -> PyResult<&'static str> {
    Ok(match selected {
        beam_runtime::cnot_search::SelectedEngine::Python => "python",
        beam_runtime::cnot_search::SelectedEngine::Rust => "rust",
    })
}

fn cnot_dispatch_capability(capabilities: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    capabilities
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing CNOT dispatch capability {key}")))?
        .extract()
}

#[pyfunction(name = "_select_cnot_dispatch")]
fn select_cnot_dispatch(
    engine: Bound<'_, PyAny>,
    max_depth: Bound<'_, PyAny>,
    strict_backend: bool,
    capabilities: Bound<'_, PyDict>,
) -> PyResult<&'static str> {
    let engine_text: String = engine
        .extract()
        .map_err(|_| PyValueError::new_err("CNOT engine must be auto, python, or rust"))?;
    let requested = match engine_text.as_str() {
        "auto" => beam_runtime::cnot_search::RequestedEngine::Auto,
        "python" => beam_runtime::cnot_search::RequestedEngine::Python,
        "rust" => beam_runtime::cnot_search::RequestedEngine::Rust,
        _ => {
            return Err(PyValueError::new_err(
                "CNOT engine must be auto, python, or rust",
            ))
        }
    };
    let depth_fits_u64 = !max_depth.call_method1("__gt__", (u64::MAX,))?.is_truthy()?;
    let caps = beam_runtime::cnot_search::DispatchCapabilities {
        native_controller: cnot_dispatch_capability(&capabilities, "native_controller")?,
        checked_control: cnot_dispatch_capability(&capabilities, "checked_control")?,
        native_stages: cnot_dispatch_capability(&capabilities, "native_stages")?,
        stream5_start: cnot_dispatch_capability(&capabilities, "stream5_start")?,
        world_exchange: cnot_dispatch_capability(&capabilities, "world_exchange")?,
        fixed_gather: cnot_dispatch_capability(&capabilities, "fixed_gather")?,
        row_gather: cnot_dispatch_capability(&capabilities, "row_gather")?,
        broadcast: cnot_dispatch_capability(&capabilities, "broadcast")?,
        compatibility_controller: cnot_dispatch_capability(
            &capabilities,
            "compatibility_controller",
        )?,
    };
    let path = beam_runtime::cnot_search::select_dispatch(
        requested,
        depth_fits_u64,
        strict_backend,
        caps,
    )
    .map_err(|error| match error {
        beam_runtime::cnot_search::DispatchError::EngineSelection(
            beam_runtime::cnot_search::EngineSelectionError::RustDepthExceedsU64,
        ) => PyValueError::new_err("CNOT Rust engine depth exceeds u64"),
        beam_runtime::cnot_search::DispatchError::StrictRequiresRustController => {
            PyRuntimeError::new_err("strict CNOT acceptance requires the Rust CNOT controller")
        }
        beam_runtime::cnot_search::DispatchError::StrictBackendUnavailable => {
            PyRuntimeError::new_err(
                "strict CNOT acceptance requires matching Rust controller, stage, transport, and Gloo ABIs on every rank",
            )
        }
        beam_runtime::cnot_search::DispatchError::RustControllerUnavailable => {
            PyRuntimeError::new_err("CNOT Rust engine requested but no compatible Rust controller is available")
        }
    })?;
    Ok(match path {
        beam_runtime::cnot_search::DispatchPath::RustController => "rust_controller",
        beam_runtime::cnot_search::DispatchPath::RustCompatibility => "rust_compatibility",
        beam_runtime::cnot_search::DispatchPath::PythonReference => "python_reference",
    })
}

fn cnot_admission_digest<'py>(
    py: Python<'py>,
    input_matrix: &Bound<'py, PyAny>,
    scorer: &Bound<'py, PyAny>,
    width: &Bound<'py, PyAny>,
    shortlist: &Bound<'py, PyAny>,
    max_depth: &Bound<'py, PyAny>,
    mode: &Bound<'py, PyAny>,
    audit: &Bound<'py, PyAny>,
    validation: &Bound<'py, PyAny>,
    selected_engine: &str,
) -> PyResult<Py<PyBytes>> {
    let builtins = py.import("builtins")?;
    let metadata = builtins
        .getattr("getattr")?
        .call1((scorer, "metadata", PyDict::new(py)))?;
    let mut fingerprints = Vec::with_capacity(5);
    for key in [
        "upstream_commit",
        "checkpoint_sha256",
        "feature_registry_sha256",
        "training_manifest_sha256",
        "training_result_sha256",
    ] {
        fingerprints.push(metadata.call_method1("get", (key,))?);
    }
    let fingerprints = PyTuple::new(py, fingerprints)?;
    let root = input_matrix.getattr("root")?;
    let shape = builtins
        .getattr("tuple")?
        .call1((root.getattr("shape")?,))?;
    let mut config = Vec::with_capacity(9);
    config.push(width.clone());
    config.push(shortlist.clone());
    config.push(max_depth.clone());
    config.push(mode.clone());
    config.push(audit.clone());
    config.push(validation.clone());
    config.push(PyString::new(py, selected_engine).into_any());
    config.push(shape.into_any());
    config.push(fingerprints.into_any());
    let config = PyTuple::new(py, config)?;
    // Preserve the Python reference's JSON encoding byte-for-byte while Rust
    // owns field order, selected-engine policy and the consensus fingerprint.
    let encoded_object = py
        .import("json")?
        .call_method1("dumps", (&config,))?
        .call_method0("encode")?;
    let encoded = encoded_object.downcast::<PyBytes>()?;
    let raw_object = input_matrix.getattr("raw")?;
    let raw = raw_object.downcast::<PyBytes>()?;
    let mut hasher = Sha256::new();
    hasher.update(raw.as_bytes());
    hasher.update(encoded.as_bytes());
    let digest = hasher.finalize();
    Ok(PyBytes::new(py, digest.as_slice()).unbind())
}

#[pymethods]
impl PyCnotMatrix {
    #[new]
    fn new(wires: u32, packed: &Bound<'_, PyBytes>) -> PyResult<Self> {
        // The Python byte slice is borrowed only for this call. beam-core owns
        // the decoded words before this method returns.
        let inner = beam_core::cnot::CnotMatrix::from_le_bytes(wires, packed.as_bytes())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    fn is_identity(&self) -> bool {
        self.inner.is_identity()
    }

    fn verifies_construction(&self, py: Python<'_>, moves: Vec<u32>) -> PyResult<bool> {
        py.allow_threads(|| self.inner.verifies_construction(&moves))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pymethods]
impl PyCnotSearchInput {
    #[getter]
    fn n(&self) -> u32 {
        self.wires
    }

    #[getter]
    fn raw(&self, py: Python<'_>) -> Py<PyBytes> {
        self.raw.clone_ref(py)
    }

    #[getter]
    fn root(&self, py: Python<'_>) -> Py<PyAny> {
        self.root.clone_ref(py)
    }

    fn is_identity(&self) -> bool {
        self.matrix.is_identity()
    }

    fn verifies_construction(&self, py: Python<'_>, moves: Vec<u32>) -> PyResult<bool> {
        py.allow_threads(|| self.matrix.verifies_construction(&moves))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

struct PyStages<'py> {
    py: Python<'py>,
    stages: Bound<'py, PyAny>,
    strict_backend: bool,
}

impl Backend for PyStages<'_> {
    type Error = PyErr;
    type Layer = Py<PyAny>;
    type Records = Py<PyAny>;

    fn root_is_goal(&mut self) -> PyResult<bool> {
        if self.stages.hasattr("input_matrix")? {
            let input_matrix = self.stages.getattr("input_matrix")?;
            if let Some(is_identity) = search_input_is_identity(&input_matrix)? {
                return Ok(is_identity);
            }
        }
        if !self.stages.hasattr("root_is_goal")? {
            return Ok(false);
        }
        self.stages.call_method0("root_is_goal")?.extract()
    }
    fn initialize(&mut self) -> PyResult<()> {
        if self.stages.hasattr("input_matrix")? {
            let input = self.stages.getattr("input_matrix")?;
            if input.is_instance_of::<PyCnotSearchInput>()
                && self.stages.hasattr("world")?
                && self.stages.hasattr("root")?
            {
                let world = self.stages.getattr("world")?;
                if world.hasattr("_checked_error_gather_control")?
                    && world.hasattr("all_ranks_support")?
                {
                    let rank: u32 = world.getattr("rank")?.extract()?;
                    let root = self.stages.getattr("root")?;
                    let parents = if rank == 0 {
                        root.clone()
                    } else {
                        let empty = self
                            .py
                            .import("builtins")?
                            .getattr("slice")?
                            .call1((self.py.None(), 0))?;
                        root.get_item(empty)?
                    };
                    self.stages.setattr("parents", parents)?;
                    let initial = if rank == 0 { vec![0_i64] } else { Vec::new() };
                    let lex = world.call_method1("longs", (initial,))?;
                    self.stages.setattr("lex", lex)?;
                    let owner_result = (|| {
                        let size = world.getattr("size")?;
                        if root.getattr("is_cuda")?.is_truthy()? {
                            let native = self.py.import("multigpubeamsearch.cnot_native")?;
                            native
                                .call_method1("cnot_owners", (&root, &size))?
                                .call_method0("item")
                        } else {
                            let size: u32 = size.extract()?;
                            crate::cnot_native_batch::cpu_owners(self.py, &root, size)?
                                .call_method0("item")
                        }
                    })();
                    let owns_root = crate::cnot_native_checked::checked_compat(
                        self.py,
                        &world,
                        "initialize CNOT domain",
                        owner_result,
                    )?
                    .call_method1("__eq__", (rank,))?
                    .is_truthy()?;
                    let closed = if owns_root {
                        root
                    } else {
                        let empty = self
                            .py
                            .import("builtins")?
                            .getattr("slice")?
                            .call1((self.py.None(), 0))?;
                        root.get_item(empty)?
                    };
                    self.stages.setattr("closed", closed)?;
                    return Ok(());
                }
            }
        }
        self.stages.call_method0("initialize")?;
        Ok(())
    }
    fn goal(&mut self, begin: u64) -> PyResult<Option<Goal>> {
        if self.stages.hasattr("input_matrix")? {
            let input = self.stages.getattr("input_matrix")?;
            if input.is_instance_of::<PyCnotSearchInput>()
                && self.stages.hasattr("world")?
                && self.stages.hasattr("parents")?
                && self.stages.hasattr("lex")?
                && self.stages.hasattr("identity")?
            {
                let world = self.stages.getattr("world")?;
                let parents = self.stages.getattr("parents")?;
                let lex = self.stages.getattr("lex")?;
                let identity = self.stages.getattr("identity")?;
                let goal = crate::cnot_native_goal::goal(
                    self.py, &world, &parents, &lex, begin, &identity,
                )?;
                return Ok(goal.map(|(slot, movement)| Goal { slot, movement }));
            }
        }
        let value = self.stages.call_method1("goal", (begin,))?;
        let result: Option<(u64, u32)> = value.extract()?;
        Ok(result.map(|(slot, movement)| Goal { slot, movement }))
    }
    fn expand(&mut self, begin: u64, k: u32) -> PyResult<()> {
        if self.stages.hasattr("input_matrix")? {
            let input = self.stages.getattr("input_matrix")?;
            if input.is_instance_of::<PyCnotSearchInput>()
                && self.stages.hasattr("world")?
                && self.stages.hasattr("parents")?
                && self.stages.hasattr("lex")?
                && self.stages.hasattr("scorer")?
                && self.stages.hasattr("mode")?
            {
                let world = self.stages.getattr("world")?;
                if world.hasattr("_checked_error_gather_control")?
                    && world.hasattr("all_ranks_support")?
                {
                    let expansion = crate::cnot_native_checked::checked_compat(
                        self.py,
                        &world,
                        "rank parent actions and expand",
                        crate::cnot_native_expand::expand(
                            self.py,
                            &self.stages,
                            begin,
                            k,
                            self.strict_backend,
                        ),
                    )?;
                    let expansion = expansion.downcast::<PyTuple>()?;
                    if expansion.len() != 2 {
                        return Err(PyValueError::new_err(
                            "CNOT expansion returned an invalid adapter result",
                        ));
                    }
                    self.stages.setattr("candidates", expansion.get_item(0)?)?;
                    self.stages.setattr("owners", expansion.get_item(1)?)?;
                    return Ok(());
                }
            }
        }
        self.stages.call_method1("expand", (begin, k))?;
        Ok(())
    }
    fn exchange_candidates(&mut self) -> PyResult<()> {
        if self.stages.hasattr("input_matrix")? {
            let input = self.stages.getattr("input_matrix")?;
            if input.is_instance_of::<PyCnotSearchInput>()
                && self.stages.hasattr("world")?
                && self.stages.hasattr("candidates")?
                && self.stages.hasattr("owners")?
            {
                let world = self.stages.getattr("world")?;
                let candidates = self.stages.getattr("candidates")?;
                let owners = self.stages.getattr("owners")?;
                let incoming = crate::cnot_native_exchange::exchange(
                    self.py,
                    &world,
                    &candidates,
                    &owners,
                    true,
                )?;
                self.stages.setattr("incoming", incoming)?;
                return Ok(());
            }
        }
        self.stages.call_method0("exchange_candidates")?;
        Ok(())
    }
    fn deduplicate_and_score(&mut self) -> PyResult<()> {
        if self.stages.hasattr("input_matrix")? {
            let input = self.stages.getattr("input_matrix")?;
            if input.is_instance_of::<PyCnotSearchInput>()
                && self.stages.hasattr("world")?
                && self.stages.hasattr("incoming")?
                && self.stages.hasattr("closed")?
                && self.stages.hasattr("scorer")?
                && self.stages.hasattr("n")?
                && self.stages.hasattr("words")?
            {
                let world = self.stages.getattr("world")?;
                if world.hasattr("_checked_error_gather_control")?
                    && world.hasattr("all_ranks_support")?
                {
                    let result = crate::cnot_native_checked::checked_compat(
                        self.py,
                        &world,
                        "deduplicate and evaluate distinct children",
                        crate::cnot_native_dedup_score::deduplicate_score(
                            self.py,
                            &self.stages,
                            self.strict_backend,
                        ),
                    )?;
                    let result = result.downcast::<PyTuple>()?;
                    if result.len() != 3 {
                        return Err(PyRuntimeError::new_err(
                            "CNOT deduplication returned an invalid adapter result",
                        ));
                    }
                    self.stages.setattr("children", result.get_item(0)?)?;
                    self.stages.setattr("meta", result.get_item(1)?)?;
                    self.stages.setattr("values", result.get_item(2)?)?;
                    return Ok(());
                }
            }
        }
        self.stages.call_method0("deduplicate_and_score")?;
        Ok(())
    }
    fn select(&mut self, width: u64) -> PyResult<()> {
        if self.stages.hasattr("input_matrix")? {
            let input = self.stages.getattr("input_matrix")?;
            if input.is_instance_of::<PyCnotSearchInput>()
                && self.stages.hasattr("world")?
                && self.stages.hasattr("values")?
                && self.stages.hasattr("meta")?
            {
                let world = self.stages.getattr("world")?;
                let values = self.stages.getattr("values")?;
                let meta = self.stages.getattr("meta")?;
                let word_keys = meta.call_method1("select", (1, 1))?;
                let result =
                    crate::cnot_native_selection::select(self.py, world, values, word_keys, width)?
                        .into_bound(self.py);
                let result = result.downcast::<PyTuple>()?;
                if result.len() != 3 {
                    return Err(PyRuntimeError::new_err(
                        "CNOT selection returned an invalid adapter result",
                    ));
                }
                self.stages.setattr("ranks", result.get_item(0)?)?;
                self.stages.setattr("local", result.get_item(1)?)?;
                self.stages.setattr("new_lex", result.get_item(2)?)?;
                return Ok(());
            }
        }
        if !self.stages.hasattr("prepare_selection")? {
            // The previous Python adapter remains usable with a newer module.
            self.stages.call_method1("select", (width,))?;
            return Ok(());
        }
        let stages = self.stages.call_method0("prepare_selection")?;
        let mut selection = PySelection { stages };
        let result = cnot_selection::select(&mut selection, width)?;
        self.stages.call_method1("commit_selection", (result,))?;
        Ok(())
    }
    fn redistribute(&mut self, depth: u64) -> PyResult<Round<Self::Layer, Self::Records>> {
        if self.stages.hasattr("input_matrix")? {
            let input = self.stages.getattr("input_matrix")?;
            if input.is_instance_of::<PyCnotSearchInput>()
                && self.stages.hasattr("world")?
                && self.stages.hasattr("ranks")?
                && self.stages.hasattr("local")?
                && self.stages.hasattr("new_lex")?
                && self.stages.hasattr("meta")?
                && self.stages.hasattr("children")?
                && self.stages.hasattr("closed")?
                && self.stages.hasattr("parents")?
                && self.stages.hasattr("candidates")?
                && self.stages.hasattr("owners")?
                && self.stages.hasattr("values")?
                && self.stages.hasattr("audit")?
                && self.stages.hasattr("n")?
                && self.stages.hasattr("words")?
            {
                let (count, layer, records) =
                    crate::cnot_native_redistribute::redistribute(self.py, &self.stages, depth)?;
                return Ok(Round {
                    count,
                    layer,
                    records,
                });
            }
        }
        let (count, layer, records) = self
            .stages
            .call_method1("redistribute", (depth,))?
            .extract()?;
        Ok(Round {
            count,
            layer,
            records,
        })
    }
    fn history_edge(
        &mut self,
        records: &Self::Records,
        target: u32,
        local_index: u64,
    ) -> PyResult<Edge> {
        if self.stages.hasattr("input_matrix")? {
            let input = self.stages.getattr("input_matrix")?;
            if input.is_instance_of::<PyCnotSearchInput>() && self.stages.hasattr("world")? {
                let world = self.stages.getattr("world")?;
                let rank: u32 = world.getattr("rank")?.extract()?;
                let prepared = (|| -> PyResult<Bound<'_, PyAny>> {
                    let edge = world.call_method1("longs", (vec![0_i64, 0, 0],))?;
                    if target == rank {
                        let record = records.bind(self.py).get_item(local_index)?;
                        edge.call_method1("copy_", (record,))?;
                    }
                    Ok(edge)
                })();
                let edge = crate::cnot_native_checked::checked_compat(
                    self.py,
                    &world,
                    "prepare CNOT history edge",
                    prepared,
                )?;
                let broadcast = world.call_method1("broadcast", (&edge, target));
                let edge = crate::cnot_native_checked::checked_compat(
                    self.py,
                    &world,
                    "broadcast CNOT history edge",
                    broadcast,
                )?;
                let parsed = crate::cnot_native_checked::checked_compat(
                    self.py,
                    &world,
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
                        Ok(PyTuple::new(
                            self.py,
                            [
                                int.call1((parent,))?,
                                int.call1((route,))?,
                                int.call1((owner,))?,
                            ],
                        )?
                        .into_any())
                    })(),
                )?;
                return Ok(Edge {
                    parent: parsed.get_item(0)?.extract()?,
                    route: parsed.get_item(1)?.extract()?,
                    owner: parsed.get_item(2)?.extract()?,
                });
            }
        }
        let (parent, route, owner) = self
            .stages
            .call_method1("history_edge", (records.bind(self.py), target, local_index))?
            .extract()?;
        Ok(Edge {
            parent,
            route,
            owner,
        })
    }
    fn verifies_construction(&mut self, gates: &[u32]) -> PyResult<bool> {
        if self.stages.hasattr("input_matrix")? {
            let input = self.stages.getattr("input_matrix")?;
            if let Some(valid) = search_input_verifies_construction(self.py, &input, gates)? {
                return Ok(valid);
            }
        }
        self.stages
            .call_method1("verifies_construction", (gates.to_vec(),))?
            .extract()
    }
    fn invalid(&self, message: &'static str) -> PyErr {
        PyRuntimeError::new_err(message)
    }
}

struct PySelection<'py> {
    stages: Bound<'py, PyAny>,
}

impl cnot_selection::Backend for PySelection<'_> {
    type Error = PyErr;
    type Output = Py<PyAny>;

    fn begin(&mut self, width: u64) -> PyResult<u64> {
        self.stages.call_method1("begin", (width,))?.extract()
    }
    fn radix_pass(&mut self, key: u32, shift: u32, remaining: u64) -> PyResult<(u64, u64)> {
        self.stages
            .call_method1("radix_pass", (key, shift, remaining))?
            .extract()
    }
    fn accept_boundary(&mut self) -> PyResult<()> {
        self.stages.call_method0("accept_boundary")?;
        Ok(())
    }
    fn finish(&mut self, count: u64, width: u64) -> PyResult<Self::Output> {
        Ok(self.stages.call_method1("finish", (count, width))?.unbind())
    }
    fn invalid(&self, message: &'static str) -> PyErr {
        PyRuntimeError::new_err(message)
    }
}

#[pyfunction]
fn run_cnot_selection(stages: Bound<'_, PyAny>, width: u64) -> PyResult<Py<PyAny>> {
    cnot_selection::select(&mut PySelection { stages }, width)
}

/// Own the CNOT rank-control path while leaving model preparation and
/// compatibility stage providers behind explicit callbacks. This keeps the
/// public Python signature intact and preserves the established collective
/// order for admission, consensus, transport selection, and stage dispatch.
#[pyfunction]
fn run_cnot_native_search<'py>(
    py: Python<'py>,
    world: Bound<'py, PyAny>,
    root: Bound<'py, PyAny>,
    scorer: Bound<'py, PyAny>,
    options: Bound<'py, PyAny>,
    callbacks: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let options = options.downcast::<PyTuple>()?;
    let callbacks = callbacks.downcast::<PyTuple>()?;
    if options.len() != 7 || callbacks.len() != 6 {
        return Err(PyRuntimeError::new_err(
            "invalid Rust CNOT search adapter contract",
        ));
    }

    let width = options.get_item(0)?;
    let shortlist = options.get_item(1)?;
    let max_depth = options.get_item(2)?;
    let mode = options.get_item(3)?;
    let audit = options.get_item(4)?;
    let validation = options.get_item(5)?;
    let engine = options.get_item(6)?;

    let input_builder = callbacks.get_item(0)?;
    let identity_builder = callbacks.get_item(1)?;
    let native_stages_builder = callbacks.get_item(2)?;
    let compatibility_stages_builder = callbacks.get_item(3)?;
    let reference_search = callbacks.get_item(4)?;
    let result_type = callbacks.get_item(5)?;

    let mut request = None;
    let admission = (|| -> PyResult<Bound<'py, PyAny>> {
        let validation_mode = validation.extract::<String>().ok();
        let input_matrix = if matches!(validation_mode.as_deref(), Some("auto") | Some("rust")) {
            let prepared = prepare_cnot_input(py, &root, &width, &shortlist, &max_depth, &mode)?;
            Py::new(py, prepared)?.into_bound(py).into_any()
        } else {
            input_builder.call1((
                root.clone(),
                width.clone(),
                shortlist.clone(),
                max_depth.clone(),
                mode.clone(),
                validation.clone(),
            ))?
        };
        let mode_text: String = mode.extract()?;
        let validation_text: String = validation.extract()?;
        let engine_text: String = engine.extract()?;
        let task = CnotRequest::new(
            input_matrix.getattr("n")?.extract()?,
            width.extract()?,
            shortlist.extract()?,
            max_depth.extract()?,
            &mode_text,
            audit.extract()?,
            &validation_text,
            &engine_text,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let rank: u32 = world.getattr("rank")?.extract()?;
        let world_size: u32 = world.getattr("size")?.extract()?;
        let request_value = task
            .bind_rank(rank, world_size)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let selected_engine = selected_engine_name(request_value.selected_engine())?;
        let digest = cnot_admission_digest(
            py,
            &input_matrix,
            &scorer,
            &width,
            &shortlist,
            &max_depth,
            &mode,
            &audit,
            &validation,
            selected_engine,
        )?;
        request = Some(request_value);
        let admitted = PyTuple::new(py, [input_matrix, digest.into_bound(py).into_any()])?;
        Ok(admitted.into_any())
    })();
    let admitted =
        crate::cnot_native_checked::checked_v2(py, &world, "validate CNOT input", admission)?;
    let admitted = admitted.downcast::<PyTuple>()?;
    if admitted.len() != 2 {
        return Err(PyRuntimeError::new_err(
            "CNOT admission returned an invalid adapter result",
        ));
    }
    let input_matrix = admitted.get_item(0)?;
    let digest: Vec<u8> = admitted.get_item(1)?.extract()?;
    let request = request.ok_or_else(|| {
        PyRuntimeError::new_err("CNOT request missing after collective admission")
    })?;
    let use_rust = request.selected_engine() == beam_runtime::cnot_search::SelectedEngine::Rust;

    let admitted_backends = crate::cnot_native_stages::preflight_strict_backend(
        py,
        &world,
        &input_matrix.getattr("root")?,
        &scorer,
        use_rust,
    )?;
    let admitted_strict = admitted_backends.borrow(py).strict;

    run_cnot_native_require_consensus(world.clone(), digest)?;
    let stream5_fingerprint = crate::cnot_stream5::run_cnot_native_start_transport(
        py,
        world.clone(),
        use_rust,
        Some(admitted_strict),
    )?;
    if admitted_strict {
        let bind_result = (|| -> PyResult<Bound<'_, PyAny>> {
            crate::cnot_native_stages::bind_stream5_admission(
                py,
                &admitted_backends,
                &world,
                stream5_fingerprint,
            )?;
            Ok(py.None().into_bound(py))
        })();
        crate::cnot_native_checked::checked_v2(
            py,
            &world,
            "bind admitted CNOT backend handles",
            bind_result,
        )?;
    }

    let wires = request.wires();
    let identity_args = PyDict::new(py);
    identity_args.set_item("device", world.getattr("device")?)?;
    let identity_result = identity_builder.call((wires,), Some(&identity_args));
    let identity =
        crate::cnot_native_checked::checked_v2(py, &world, "build CNOT identity", identity_result)?;
    let alphabet = wires * (wires - 1);

    if use_rust {
        let native_stages_result = native_stages_builder.call0();
        let native_stages = crate::cnot_native_checked::checked_v2(
            py,
            &world,
            "load CNOT native stage runner",
            native_stages_result,
        )?;
        let supports_native = crate::cnot_native_stages::run_cnot_native_all_ranks_support(
            world.clone(),
            !native_stages.is_none(),
        )?;
        if admitted_strict && !supports_native {
            return Err(PyRuntimeError::new_err(
                "strict CNOT acceptance requires the Rust native stage runner on every rank",
            ));
        }
        let width = request.width();
        let shortlist_value = request.shortlist();
        let depth = request.max_depth();
        let mode = PyString::new(py, request.mode().as_str()).into_any();

        if supports_native {
            let native_root = input_matrix.getattr("root")?;
            let topology_kwargs = crate::cnot_native_checked::checked_v2(
                py,
                &world,
                "prepare admitted CNOT rank topology",
                (|| -> PyResult<Bound<'_, PyAny>> {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("admitted_topology", (request.rank(), request.world()))?;
                    kwargs.set_item(
                        "ranked_request",
                        Py::new(py, PyCnotRankedRequest { inner: request })?,
                    )?;
                    Ok(kwargs.into_any())
                })(),
            )?;
            let topology_kwargs = topology_kwargs.downcast::<PyDict>()?;
            return native_stages.call(
                (
                    world,
                    native_root,
                    identity,
                    input_matrix,
                    scorer,
                    mode,
                    audit,
                    width,
                    shortlist_value,
                    alphabet,
                    depth,
                    admitted_backends,
                ),
                Some(&topology_kwargs),
            );
        }

        let compatibility_stages = compatibility_stages_builder.call1((
            world.clone(),
            input_matrix.getattr("root")?,
            identity,
            input_matrix,
            scorer,
            mode,
            audit,
        ));
        let stages = crate::cnot_native_checked::checked_v2(
            py,
            &world,
            "build CNOT compatibility stages",
            compatibility_stages,
        )?;
        return Ok(
            run_cnot_with_options(py, stages, request.search_options(), admitted_strict)?
                .into_bound(py),
        );
    }

    let admitted_root = input_matrix.getattr("root")?;
    if py
        .import("torch")?
        .call_method1("equal", (&admitted_root, &identity))?
        .extract::<bool>()?
    {
        return result_type.call1((
            "solved",
            PyTuple::empty(py),
            PyList::empty(py),
            PyList::empty(py),
        ));
    }
    reference_search.call1((
        world,
        admitted_root,
        identity,
        input_matrix,
        scorer,
        width,
        shortlist,
        max_depth,
        mode,
        audit,
    ))
}

/// Run on the calling rank's thread: Torch's current device/stream and
/// inference-mode context are thread-local. Only complete stages cross PyO3.
#[pyfunction]
fn run_cnot<'py>(
    py: Python<'py>,
    stages: Bound<'py, PyAny>,
    rank: u32,
    world: u32,
    width: u64,
    shortlist: u32,
    alphabet: u32,
    max_depth: u64,
) -> PyResult<Py<PyAny>> {
    let strict_backend = matches!(
        std::env::var("MGBS_CNOT_STRICT_BACKEND").as_deref(),
        Ok("1")
    );
    run_cnot_with_options(
        py,
        stages,
        Options {
            rank,
            world,
            width,
            shortlist,
            alphabet,
            max_depth,
        },
        strict_backend,
    )
}

fn run_cnot_with_options<'py>(
    py: Python<'py>,
    stages: Bound<'py, PyAny>,
    options: Options,
    strict_backend: bool,
) -> PyResult<Py<PyAny>> {
    let mut backend = PyStages {
        py,
        stages,
        strict_backend,
    };
    let outcome = cnot_search::search(&mut backend, options)?;
    outcome_to_python(py, outcome)
}

pub(crate) fn outcome_to_python(
    py: Python<'_>,
    outcome: cnot_search::Outcome<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let gates = match outcome.gates {
        Some(gates) => PyTuple::new(py, gates)?.into_any().unbind(),
        None => py.None(),
    };
    let layers = PyList::new(py, outcome.layers)?;
    let routes = PyList::empty(py);
    for route in outcome.routes {
        let item = PyDict::new(py);
        item.set_item("source", route.source)?;
        item.set_item("hash_owner", route.hash_owner)?;
        item.set_item("target", route.target)?;
        item.set_item("move", route.movement)?;
        routes.append(item)?;
    }
    Ok(py
        .import("multigpubeamsearch.cnot")?
        .getattr("CnotResult")?
        .call1((outcome.status, gates, layers, routes))?
        .unbind())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCnotMatrix>()?;
    module.add_class::<PyCnotSearchInput>()?;
    module.add_class::<PyCnotRankedRequest>()?;
    module.add_function(wrap_pyfunction!(prepare_cnot_input, module)?)?;
    module.add_function(wrap_pyfunction!(validate_cnot_search_controls, module)?)?;
    module.add_function(wrap_pyfunction!(cnot_public_dispatch_digest, module)?)?;
    module.add_function(wrap_pyfunction!(run_cnot_native_require_consensus, module)?)?;
    module.add_function(wrap_pyfunction!(select_cnot_dispatch, module)?)?;
    module.add_function(wrap_pyfunction!(run_cnot_native_search, module)?)?;
    module.add_function(wrap_pyfunction!(run_cnot, module)?)?;
    module.add_function(wrap_pyfunction!(run_cnot_selection, module)?)
}
