//! Thin Python adapter; runtime implementation lives in beam-runtime.
#![forbid(unsafe_code)]
mod inference;
mod pipeline;
mod planning;
mod probes;
mod search_session;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
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

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pyclass]
struct ResidentWorkers {
    inner: Mutex<beam_runtime::ResidentWorkers>,
    cancellation: Arc<AtomicBool>,
}
#[pymethods]
impl ResidentWorkers {
    #[new]
    fn new(
        py: Python<'_>,
        command: Vec<String>,
        world: u32,
        directory: PathBuf,
        cwd: PathBuf,
        env: BTreeMap<String, String>,
    ) -> PyResult<Self> {
        let group = py
            .allow_threads(move || {
                beam_runtime::ResidentWorkers::launch(&command, world, &directory, &cwd, &env)
            })
            .map_err(runtime_error)?;
        let cancellation = group.cancellation();
        Ok(Self {
            inner: Mutex::new(group),
            cancellation,
        })
    }
    fn submit(
        &self,
        py: Python<'_>,
        start: Vec<u8>,
        run_dir: PathBuf,
        timeout: f64,
    ) -> PyResult<f64> {
        if !timeout.is_finite() || timeout <= 0.0 {
            return Err(PyValueError::new_err("timeout must be finite and positive"));
        }
        let timeout = Duration::try_from_secs_f64(timeout)
            .map_err(|_| PyValueError::new_err("timeout overflow"))?;
        py.allow_threads(|| {
            self.inner
                .lock()
                .map_err(runtime_error)?
                .submit(&start, &run_dir, timeout)
                .map(|elapsed| elapsed.as_secs_f64())
                .map_err(runtime_error)
        })
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.cancellation.store(true, Ordering::Release);
        py.allow_threads(|| {
            self.inner
                .lock()
                .map_err(runtime_error)?
                .close()
                .map_err(runtime_error)
        })
    }
    fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        self.cancellation.store(true, Ordering::Release);
        py.allow_threads(|| {
            // Active submit observes the token and closes its owned group. An
            // idle session can be closed immediately without waiting on a job.
            if let Ok(mut inner) = self.inner.try_lock() {
                inner.close().map_err(runtime_error)?;
            }
            Ok(())
        })
    }
    #[getter]
    fn pids(&self) -> PyResult<Vec<u32>> {
        Ok(self.inner.lock().map_err(runtime_error)?.pids())
    }
    #[getter]
    fn sequence(&self) -> PyResult<u64> {
        Ok(self.inner.lock().map_err(runtime_error)?.sequence())
    }
    #[getter]
    fn closed(&self) -> PyResult<bool> {
        Ok(self.inner.lock().map_err(runtime_error)?.is_closed())
    }
    fn exit_codes(&self) -> PyResult<Vec<Option<i32>>> {
        self.inner
            .lock()
            .map_err(runtime_error)?
            .exit_codes()
            .map_err(runtime_error)
    }
}
#[pymodule]
fn _beam_runtime(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ResidentWorkers>()?;
    probes::register(module)?;
    planning::register(module)?;
    pipeline::register(module)?;
    inference::register(module)?;
    search_session::register(module)?;
    module.add("ABI_VERSION", 1)?;
    Ok(())
}
