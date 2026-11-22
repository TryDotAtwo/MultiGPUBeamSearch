use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
fn error(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}
fn duration(seconds: f64) -> PyResult<Duration> {
    if !seconds.is_finite() || seconds <= 0. {
        return Err(PyValueError::new_err("timeout must be finite and positive"));
    }
    Duration::try_from_secs_f64(seconds).map_err(error)
}
#[pyclass]
struct InferencePool {
    inner: Mutex<beam_runtime::inference::InferencePool>,
    cancel: Arc<AtomicBool>,
}
#[pymethods]
impl InferencePool {
    #[new]
    fn new(py: Python<'_>, config: &str, timeout: f64) -> PyResult<Self> {
        let config =
            serde_json::from_str(config).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let timeout = duration(timeout)?;
        let inner = py
            .allow_threads(|| beam_runtime::inference::InferencePool::new(config, timeout))
            .map_err(error)?;
        Ok(Self {
            cancel: inner.cancellation(),
            inner: Mutex::new(inner),
        })
    }
    fn measure(&self, py: Python<'_>, batch: u64, parents: u64, timeout: f64) -> PyResult<String> {
        let timeout = duration(timeout)?;
        py.allow_threads(|| {
            let rows = self
                .inner
                .lock()
                .map_err(error)?
                .measure(batch, parents, timeout)
                .map_err(error)?;
            serde_json::to_string(&rows).map_err(error)
        })
    }
    fn tune(&self, py: Python<'_>, config: &str, timeout: f64) -> PyResult<String> {
        let config =
            serde_json::from_str(config).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let timeout = duration(timeout)?;
        py.allow_threads(|| {
            let selection = self
                .inner
                .lock()
                .map_err(error)?
                .tune(&config, timeout)
                .map_err(error)?;
            serde_json::to_string(&selection).map_err(error)
        })
    }
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.cancel();
        py.allow_threads(|| self.inner.lock().map_err(error)?.close().map_err(error))
    }
    #[getter]
    fn pids(&self) -> PyResult<Vec<Option<u32>>> {
        Ok(self.inner.lock().map_err(error)?.pids())
    }
    #[getter]
    fn starts(&self) -> PyResult<u64> {
        Ok(self.inner.lock().map_err(error)?.starts())
    }
    #[getter]
    fn capacity(&self) -> PyResult<u64> {
        Ok(self.inner.lock().map_err(error)?.capacity())
    }
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<InferencePool>()?;
    Ok(())
}
