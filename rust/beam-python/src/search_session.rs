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
        py.allow_threads(|| {
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
    module.add_class::<SearchSession>()?;
    module.add_function(wrap_pyfunction!(parse_terminal, module)?)?;
    Ok(())
}
