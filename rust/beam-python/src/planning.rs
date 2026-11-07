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
pyo3::create_exception!(_beam_runtime, CapacityRejected, PyValueError);

fn error(e: beam_runtime::Error) -> PyErr {
    match e {
        beam_runtime::Error::CapacityRejected(s) => CapacityRejected::new_err(s),
        beam_runtime::Error::Contract(e) => PyValueError::new_err(e.to_string()),
        e if e.to_string().contains("timed out") => PyTimeoutError::new_err(e.to_string()),
        e => PyRuntimeError::new_err(e.to_string()),
    }
}
fn duration(value: f64) -> PyResult<Duration> {
    if !value.is_finite() || value <= 0. {
        return Err(PyValueError::new_err("timeout must be finite and positive"));
    }
    Duration::try_from_secs_f64(value).map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyclass]
struct PlanningSession {
    inner: Mutex<beam_runtime::planning::PlanningSession>,
    cancellation: Arc<AtomicBool>,
}
#[pymethods]
impl PlanningSession {
    #[new]
    fn new(
        py: Python<'_>,
        command: Vec<String>,
        environment: BTreeMap<String, String>,
        world: u32,
        cwd: PathBuf,
        directory: PathBuf,
        timeout: f64,
    ) -> PyResult<Self> {
        let timeout = duration(timeout)?;
        let inner = py
            .allow_threads(|| {
                beam_runtime::planning::PlanningSession::launch(
                    &command,
                    &environment,
                    world,
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
    fn admit(
        &self,
        py: Python<'_>,
        beam: u64,
        environment: BTreeMap<String, String>,
    ) -> PyResult<String> {
        py.allow_threads(|| {
            let rows = self
                .inner
                .lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .admit(beam, &environment)
                .map_err(error)?;
            serde_json::to_string(&rows).map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
    fn measure_components(
        &self,
        py: Python<'_>,
        beam: u64,
        environment: BTreeMap<String, String>,
    ) -> PyResult<String> {
        py.allow_threads(|| {
            let rows = self
                .inner
                .lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .measure_components(beam, &environment)
                .map_err(error)?;
            serde_json::to_string(&rows).map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
    fn set_deadline(&self, py: Python<'_>, timeout: f64) -> PyResult<()> {
        let timeout = duration(timeout)?;
        py.allow_threads(|| {
            self.inner
                .lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .set_deadline(timeout)
                .map_err(error)
        })
    }
    #[getter]
    fn supports_components(&self) -> PyResult<bool> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .supports_components())
    }
    #[getter]
    fn pids(&self) -> PyResult<Vec<u32>> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .pids())
    }
    fn exit_codes(&self) -> PyResult<Vec<Option<i32>>> {
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .exit_codes()
            .map_err(error)
    }
    fn cancel(&self) {
        self.cancellation.store(true, Ordering::Release);
    }
    fn find_capacity(
        &self,
        py: Python<'_>,
        profiles: &str,
        upper: u64,
        alignment: u64,
    ) -> PyResult<String> {
        let profiles: Vec<BTreeMap<String, String>> =
            serde_json::from_str(profiles).map_err(|e| PyValueError::new_err(e.to_string()))?;
        py.allow_threads(|| {
            let result = self
                .inner
                .lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .find_capacity(&profiles, upper, alignment)
                .map_err(error)?;
            serde_json::to_string(&result).map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.cancel();
        py.allow_threads(|| {
            self.inner
                .lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .close()
                .map_err(error)
        })
    }
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PlanningSession>()?;
    module.add(
        "CapacityRejected",
        module.py().get_type::<CapacityRejected>(),
    )?;
    module.add_function(wrap_pyfunction!(select_profile, module)?)?;
    module.add_function(wrap_pyfunction!(promote_optimization, module)?)?;
    Ok(())
}

#[pyfunction]
fn select_profile(
    py: Python<'_>,
    samples: &str,
    baseline: &str,
    min_improvement: f64,
) -> PyResult<String> {
    let rows: Vec<beam_core::calibration::Measurement> =
        serde_json::from_str(samples).map_err(|e| PyValueError::new_err(e.to_string()))?;
    py.allow_threads(|| {
        let (estimate,rejected)=beam_core::calibration::select(&rows,baseline,min_improvement).map_err(|e| PyValueError::new_err(e.to_string()))?;
        serde_json::to_string(&serde_json::json!({"estimate":estimate,"rejected":rejected,"statistics_policy":"bootstrap-2000-splitmix64-v1"}))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}
#[pyfunction]
fn promote_optimization(py: Python<'_>, pairs: &str) -> PyResult<String> {
    let rows: Vec<beam_core::calibration::PairedMeasurement> =
        serde_json::from_str(pairs).map_err(|e| PyValueError::new_err(e.to_string()))?;
    py.allow_threads(|| {
        let decision = beam_core::calibration::paired_promotion(&rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        serde_json::to_string(&decision).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}
