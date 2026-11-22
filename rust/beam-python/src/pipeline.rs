use beam_runtime::calibration::ProfileEvaluator;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{
    collections::BTreeMap,
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
struct PipelineProbe {
    inner: Mutex<beam_runtime::pipeline::PipelineProbe>,
    cancel: Arc<AtomicBool>,
}
#[pymethods]
impl PipelineProbe {
    #[new]
    fn new(py: Python<'_>, config: &str, timeout: f64) -> PyResult<Self> {
        let config =
            serde_json::from_str(config).map_err(|e| PyValueError::new_err(e.to_string()))?;
        if !timeout.is_finite() || timeout <= 0. {
            return Err(PyValueError::new_err("timeout must be finite and positive"));
        }
        let timeout = Duration::try_from_secs_f64(timeout).map_err(error)?;
        let inner = py
            .allow_threads(|| beam_runtime::pipeline::PipelineProbe::launch(config, timeout))
            .map_err(error)?;
        Ok(Self {
            cancel: inner.cancellation(),
            inner: Mutex::new(inner),
        })
    }
    fn admit(&self, py: Python<'_>, environment: BTreeMap<String, String>) -> PyResult<String> {
        py.allow_threads(|| {
            let rows = self
                .inner
                .lock()
                .map_err(error)?
                .admit(&environment)
                .map_err(error)?;
            serde_json::to_string(&rows).map_err(error)
        })
    }
    fn finish_planning(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner
                .lock()
                .map_err(error)?
                .finish_planning()
                .map_err(error)
        })
    }
    fn measure(
        &self,
        py: Python<'_>,
        environment: BTreeMap<String, String>,
        plans: &str,
    ) -> PyResult<String> {
        let plans: Vec<serde_json::Value> = serde_json::from_str(plans).map_err(error)?;
        py.allow_threads(|| {
            let rows = self
                .inner
                .lock()
                .map_err(error)?
                .measure(&environment, &plans)
                .map_err(error)?;
            serde_json::to_string(&rows).map_err(error)
        })
    }
    fn tune(&self, py: Python<'_>, profiles: &str, micro: u64) -> PyResult<String> {
        let profiles: Vec<beam_runtime::calibration::Profile> =
            serde_json::from_str(profiles).map_err(error)?;
        py.allow_threads(|| {
            let selection = self
                .inner
                .lock()
                .map_err(error)?
                .tune(&profiles, micro)
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
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PipelineProbe>()?;
    Ok(())
}
