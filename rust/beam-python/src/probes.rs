use pyo3::{
    exceptions::{PyRuntimeError, PyTimeoutError, PyValueError},
    prelude::*,
};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

fn error(error: impl std::fmt::Display) -> PyErr {
    let text = error.to_string();
    if text.contains("timed out") {
        PyTimeoutError::new_err(text)
    } else {
        PyRuntimeError::new_err(text)
    }
}
fn duration(seconds: f64) -> PyResult<Duration> {
    if !seconds.is_finite() || seconds <= 0. {
        return Err(PyValueError::new_err("timeout must be finite and positive"));
    }
    Duration::try_from_secs_f64(seconds).map_err(error)
}
#[pyclass]
struct ProbeSession {
    inner: Mutex<beam_runtime::probes::ProbeSession>,
    cancellation: Arc<AtomicBool>,
}
#[pymethods]
impl ProbeSession {
    #[new]
    fn new(
        py: Python<'_>,
        command: Vec<String>,
        cwd: PathBuf,
        env: BTreeMap<String, String>,
        log: PathBuf,
        timeout: f64,
    ) -> PyResult<Self> {
        let timeout = duration(timeout)?;
        let inner = py
            .allow_threads(|| {
                beam_runtime::probes::ProbeSession::launch(&command, &cwd, &env, &log, timeout)
            })
            .map_err(error)?;
        Ok(Self {
            cancellation: inner.cancellation(),
            inner: Mutex::new(inner),
        })
    }
    fn send_request(&self, py: Python<'_>, request: &str) -> PyResult<()> {
        let value = serde_json::from_str(request).map_err(error)?;
        py.allow_threads(|| {
            self.inner
                .lock()
                .map_err(error)?
                .send(&value)
                .map_err(error)
        })
    }
    fn receive(&self, py: Python<'_>, timeout: f64) -> PyResult<String> {
        let timeout = duration(timeout)?;
        py.allow_threads(|| {
            let value = self
                .inner
                .lock()
                .map_err(error)?
                .receive(timeout)
                .map_err(error)?;
            serde_json::to_string(&value).map_err(error)
        })
    }
    fn set_deadline(&self, timeout: f64) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(error)?
            .set_deadline(duration(timeout)?)
            .map_err(error)
    }
    #[getter]
    fn pid(&self) -> PyResult<u32> {
        Ok(self.inner.lock().map_err(error)?.pid())
    }
    fn exit_code(&self) -> PyResult<Option<i32>> {
        self.inner.lock().map_err(error)?.exit_code().map_err(error)
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.cancellation.store(true, Ordering::Release);
        py.allow_threads(|| self.inner.lock().map_err(error)?.close().map_err(error))
    }
    fn cancel(&self) {
        self.cancellation.store(true, Ordering::Release);
    }
}

#[pyclass]
struct ProbeCohort {
    inner: Mutex<beam_runtime::probes::ProbeCohort>,
    cancellation: Arc<AtomicBool>,
}
#[pymethods]
impl ProbeCohort {
    #[new]
    fn new(
        py: Python<'_>,
        commands: Vec<Vec<String>>,
        environments: Vec<BTreeMap<String, String>>,
        cwd: PathBuf,
        directory: PathBuf,
        timeout: f64,
    ) -> PyResult<Self> {
        let timeout = duration(timeout)?;
        let inner = py
            .allow_threads(|| {
                beam_runtime::probes::ProbeCohort::launch(
                    &commands,
                    &environments,
                    &cwd,
                    &directory,
                    timeout,
                )
            })
            .map_err(error)?;
        Ok(Self {
            cancellation: inner.cancellation(),
            inner: Mutex::new(inner),
        })
    }
    fn send_requests(&self, py: Python<'_>, requests: &str) -> PyResult<()> {
        let requests = serde_json::from_str::<Vec<serde_json::Value>>(requests).map_err(error)?;
        py.allow_threads(|| {
            self.inner
                .lock()
                .map_err(error)?
                .send(&requests)
                .map_err(error)
        })
    }
    fn collect(&self, py: Python<'_>, timeout: f64) -> PyResult<String> {
        let timeout = duration(timeout)?;
        py.allow_threads(|| {
            let values = self
                .inner
                .lock()
                .map_err(error)?
                .collect(timeout)
                .map_err(error)?;
            serde_json::to_string(&values).map_err(error)
        })
    }
    fn set_deadline(&self, timeout: f64) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(error)?
            .set_deadline(duration(timeout)?)
            .map_err(error)
    }
    #[getter]
    fn pids(&self) -> PyResult<Vec<u32>> {
        Ok(self.inner.lock().map_err(error)?.pids())
    }
    fn exit_codes(&self) -> PyResult<Vec<Option<i32>>> {
        self.inner
            .lock()
            .map_err(error)?
            .exit_codes()
            .map_err(error)
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.cancellation.store(true, Ordering::Release);
        py.allow_threads(|| self.inner.lock().map_err(error)?.close().map_err(error))
    }
    fn cancel(&self) {
        self.cancellation.store(true, Ordering::Release);
    }
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ProbeSession>()?;
    module.add_class::<ProbeCohort>()?;
    Ok(())
}
