use beam_core::search::PermutationGraph;
use beam_runtime::search_session::{PreparationIdentity, SessionIdentity};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
fn error(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}
#[pyfunction]
#[pyo3(signature=(center, generators, graph_hash, model_artifact_hash, model_backend, devices, beam_width, max_steps, runtime_build, options, cuda_visible_devices, cuda_device_order))]
fn build_session_identity(
    center: Vec<u8>,
    generators: Vec<Vec<u32>>,
    graph_hash: String,
    model_artifact_hash: String,
    model_backend: String,
    devices: Vec<u32>,
    beam_width: u64,
    max_steps: u32,
    runtime_build: String,
    options: String,
    cuda_visible_devices: Option<String>,
    cuda_device_order: Option<String>,
) -> PyResult<String> {
    let runtime_build: serde_json::Value = serde_json::from_str(&runtime_build).map_err(error)?;
    let options: serde_json::Value = serde_json::from_str(&options).map_err(error)?;
    let identity = SessionIdentity::new(
        PermutationGraph { center, generators },
        graph_hash,
        model_artifact_hash,
        model_backend,
        runtime_build,
        devices,
        beam_width,
        max_steps,
        options,
        cuda_visible_devices,
        cuda_device_order,
    );
    serde_json::to_string(&identity).map_err(error)
}
#[pyfunction]
#[pyo3(signature=(graph_hash, geometry, center, generators, model_artifact_hash, artifact_sizes, model_backend, devices, binding_scope, beam_width, max_steps, options_repr, cuda_visible_devices, cuda_device_order, runtime_build))]
fn build_preparation_identity(
    graph_hash: Option<String>,
    geometry: Vec<Option<u64>>,
    center: Option<Vec<u32>>,
    generators: Option<Vec<Vec<u32>>>,
    model_artifact_hash: String,
    artifact_sizes: Option<Vec<u64>>,
    model_backend: String,
    devices: Vec<u32>,
    binding_scope: bool,
    beam_width: u64,
    max_steps: u32,
    options_repr: String,
    cuda_visible_devices: Option<String>,
    cuda_device_order: Option<String>,
    runtime_build: Option<String>,
) -> PyResult<String> {
    let geometry: [Option<u64>; 3] = geometry.try_into().map_err(|_| {
        PyValueError::new_err("preparation geometry must contain exactly three values")
    })?;
    let runtime_build: Option<serde_json::Value> = runtime_build
        .map(|raw| serde_json::from_str(&raw))
        .transpose()
        .map_err(error)?;
    let identity = PreparationIdentity::new(
        graph_hash,
        geometry,
        center,
        generators,
        model_artifact_hash,
        artifact_sizes,
        model_backend,
        devices,
        binding_scope,
        beam_width,
        max_steps,
        options_repr,
        cuda_visible_devices,
        cuda_device_order,
        runtime_build,
    );
    serde_json::to_string(&identity).map_err(error)
}
#[pyclass]
struct SearchSession {
    inner: Mutex<beam_runtime::search_session::SearchSession>,
    cancel: Arc<AtomicBool>,
}
#[pymethods]
impl SearchSession {
    #[new]
    fn new() -> Self {
        let inner = beam_runtime::search_session::SearchSession::new();
        Self {
            cancel: inner.cancellation(),
            inner: Mutex::new(inner),
        }
    }
    fn run(
        &self,
        py: Python<'_>,
        identity: &str,
        launch: &str,
        start: Vec<u8>,
        run_dir: PathBuf,
        timeout: f64,
        metadata: &str,
    ) -> PyResult<String> {
        if !timeout.is_finite() || timeout <= 0. {
            return Err(PyValueError::new_err("timeout must be finite and positive"));
        }
        let timeout = Duration::try_from_secs_f64(timeout).map_err(error)?;
        let identity = serde_json::from_str(identity).map_err(error)?;
        let launch = serde_json::from_str(launch).map_err(error)?;
        let metadata = serde_json::from_str(metadata).map_err(error)?;
        crate::interrupt::run(py, &self.cancel, || {
            let mut inner = self.inner.try_lock().map_err(|_| {
                PyRuntimeError::new_err("concurrent native session searches are unsupported")
            })?;
            let result = inner
                .run(&identity, &launch, &start, &run_dir, timeout, &metadata)
                .map_err(error)?;
            serde_json::to_string(&result).map_err(error)
        })
    }
    fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        self.cancel.store(true, Ordering::Release);
        py.allow_threads(|| {
            if let Ok(mut inner) = self.inner.try_lock() {
                inner.close().map_err(error)?;
            }
            Ok(())
        })
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.cancel.store(true, Ordering::Release);
        py.allow_threads(|| self.inner.lock().map_err(error)?.close().map_err(error))
    }
    fn begin_preparation(&self, py: Python<'_>, identity: &str) -> PyResult<bool> {
        let identity: PreparationIdentity = serde_json::from_str(identity).map_err(error)?;
        py.allow_threads(|| {
            let mut inner = self.inner.try_lock().map_err(|_| {
                PyRuntimeError::new_err("concurrent native session searches are unsupported")
            })?;
            inner.begin_preparation(&identity).map_err(error)
        })
    }
    fn preparation_matches(&self, identity: &str) -> PyResult<bool> {
        let inner = self.inner.try_lock().map_err(|_| {
            PyRuntimeError::new_err("concurrent native session searches are unsupported")
        })?;
        let identity: PreparationIdentity = serde_json::from_str(identity).map_err(error)?;
        inner.preparation_matches(&identity).map_err(error)
    }
    fn remember_preparation(&self, identity: &str) -> PyResult<()> {
        let mut inner = self.inner.try_lock().map_err(|_| {
            PyRuntimeError::new_err("concurrent native session searches are unsupported")
        })?;
        let identity: PreparationIdentity = serde_json::from_str(identity).map_err(error)?;
        inner.remember_preparation(&identity).map_err(error)
    }
    fn remember_preparation_result(&self, identity: &str, metadata: &str) -> PyResult<bool> {
        let identity: PreparationIdentity = serde_json::from_str(identity).map_err(error)?;
        let metadata: serde_json::Value = serde_json::from_str(metadata).map_err(error)?;
        let mut inner = self.inner.try_lock().map_err(|_| {
            PyRuntimeError::new_err("concurrent native session searches are unsupported")
        })?;
        inner
            .remember_preparation_result(&identity, &metadata)
            .map_err(error)
    }
    #[getter]
    fn pids(&self) -> PyResult<Vec<u32>> {
        Ok(self.inner.lock().map_err(error)?.pids())
    }
    #[getter]
    fn closed(&self) -> PyResult<bool> {
        Ok(self.inner.lock().map_err(error)?.is_closed())
    }
}
#[pyfunction]
#[pyo3(signature=(path,graph,start,max_steps=None))]
fn parse_terminal(
    py: Python<'_>,
    path: PathBuf,
    graph: &str,
    start: Vec<u8>,
    max_steps: Option<u32>,
) -> PyResult<String> {
    let graph = serde_json::from_str(graph).map_err(error)?;
    py.allow_threads(|| {
        let result = beam_runtime::search_session::parse_terminal(&path, &graph, &start, max_steps)
            .map_err(error)?;
        serde_json::to_string(&result).map_err(error)
    })
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("SEARCH_TERMINAL_RESULT_VERSION", 1)?;
    module.add_function(wrap_pyfunction!(build_preparation_identity, module)?)?;
    module.add_function(wrap_pyfunction!(build_session_identity, module)?)?;
    module.add_class::<SearchSession>()?;
    module.add_function(wrap_pyfunction!(parse_terminal, module)?)?;
    Ok(())
}
