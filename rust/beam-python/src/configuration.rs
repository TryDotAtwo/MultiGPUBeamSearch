use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

#[pyfunction]
fn native_depth_budget(max_steps: u64, touch_bfs_radius: u32) -> (u64, u32) {
    beam_core::search::native_depth_budget(max_steps, touch_bfs_radius)
}

#[pyfunction]
fn touch_bfs_worst_case_entries(move_count: u32, radius: u32, max_entries: u64) -> PyResult<u64> {
    beam_core::search::touch_bfs_worst_case_entries(move_count, radius, max_entries)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn validate_options(options: &str) -> PyResult<String> {
    let options: beam_core::config::NativeOptions =
        serde_json::from_str(options).map_err(|e| PyValueError::new_err(e.to_string()))?;
    options
        .validate()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&options).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn validate_search_controls(request: &str) -> PyResult<(String, String)> {
    let controls: beam_core::search::PublicSearchControls =
        serde_json::from_str(request).map_err(|e| PyValueError::new_err(e.to_string()))?;
    match beam_core::search::validate_public_search_controls(&controls) {
        Ok(()) => Ok(("valid".into(), String::new())),
        Err(beam_core::search::SearchControlError::Invalid(message)) => {
            Ok(("invalid".into(), message))
        }
        Err(beam_core::search::SearchControlError::Unsupported(message)) => {
            Ok(("unsupported".into(), message))
        }
    }
}

#[pyfunction]
fn build_graph_contract(request: &str) -> PyResult<String> {
    let request: beam_core::search::PublicGraphContractRequest =
        serde_json::from_str(request).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let contract = beam_core::search::build_public_graph_contract(request)
        .map_err(|message| PyValueError::new_err(message))?;
    serde_json::to_string(&contract).map_err(|error| PyValueError::new_err(error.to_string()))
}
#[pyfunction]
fn inference_coarse_batches(cap: u64) -> PyResult<Vec<u64>> {
    beam_core::inference::coarse_batches(cap)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn inference_refinement_batches(winner: u64, cap: u64, visited: Vec<u64>) -> PyResult<Vec<u64>> {
    let visited = visited
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    beam_core::inference::refinement_batches(winner, cap, &visited)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}
#[pyfunction]
fn inference_probe_time_allowance(records: Vec<Vec<f64>>, default: f64) -> PyResult<f64> {
    beam_core::inference::probe_time_allowance(&records, default)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn inference_capacity_cache_reusable(previous: &str, current: &str) -> PyResult<bool> {
    let previous: Option<std::collections::BTreeMap<String, u64>> =
        serde_json::from_str(previous).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let current: Option<std::collections::BTreeMap<String, u64>> =
        serde_json::from_str(current).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(beam_core::inference::capacity_cache_reusable(
        previous.as_ref(),
        current.as_ref(),
    ))
}

#[pyfunction]
fn inference_select_compatible(request: &str) -> PyResult<String> {
    let request: serde_json::Value =
        serde_json::from_str(request).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let baseline = request
        .get("baseline")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| PyValueError::new_err("missing inference baseline"))?;
    let samples = request
        .get("samples")
        .cloned()
        .ok_or_else(|| PyValueError::new_err("missing inference samples"))?;
    let samples: Vec<beam_core::calibration::Measurement> = serde_json::from_value(samples)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let selected = beam_core::inference::select_inference_compatible(&samples, baseline)
        .map_err(PyValueError::new_err)?;
    serde_json::to_string(&selected).map_err(|error| PyValueError::new_err(error.to_string()))
}
#[pyclass]
struct InferenceCalibration {
    request: beam_runtime::inference_job::Request,
    cancel: Arc<AtomicBool>,
    running: Mutex<()>,
}
#[pyclass]
struct DownstreamCalibration {
    request: beam_runtime::downstream_job::Request,
    cancel: Arc<AtomicBool>,
    running: Mutex<()>,
}
#[pyclass]
struct FrontierCalibration {
    request: beam_runtime::frontier_job::Request,
    cancel: Arc<AtomicBool>,
    running: Mutex<()>,
}
#[pymethods]
impl FrontierCalibration {
    #[new]
    fn new(request: &str) -> PyResult<Self> {
        let request =
            serde_json::from_str(request).map_err(|e| PyValueError::new_err(e.to_string()))?;
        beam_runtime::frontier_job::validate(&request)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            request,
            cancel: Arc::new(AtomicBool::new(false)),
            running: Mutex::new(()),
        })
    }
    fn run(&self, py: Python<'_>) -> PyResult<String> {
        crate::interrupt::run(py, &self.cancel, || {
            let _guard = self.running.try_lock().map_err(|_| {
                PyRuntimeError::new_err("concurrent frontier calibration is unsupported")
            })?;
            let result = beam_runtime::frontier_job::run(&self.request, self.cancel.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            serde_json::to_string(&result).map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
}
#[pymethods]
impl DownstreamCalibration {
    #[new]
    fn new(request: &str) -> PyResult<Self> {
        let request =
            serde_json::from_str(request).map_err(|e| PyValueError::new_err(e.to_string()))?;
        beam_runtime::downstream_job::prepare(&request)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            request,
            cancel: Arc::new(AtomicBool::new(false)),
            running: Mutex::new(()),
        })
    }
    fn run(&self, py: Python<'_>) -> PyResult<String> {
        crate::interrupt::run(py, &self.cancel, || {
            let _guard = self.running.try_lock().map_err(|_| {
                PyRuntimeError::new_err("concurrent downstream calibration is unsupported")
            })?;
            let result = beam_runtime::downstream_job::run(&self.request, self.cancel.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            serde_json::to_string(&result).map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
}
#[pymethods]
impl InferenceCalibration {
    #[new]
    fn new(request: &str) -> PyResult<Self> {
        let request =
            serde_json::from_str(request).map_err(|e| PyValueError::new_err(e.to_string()))?;
        beam_runtime::inference_job::prepare(&request)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            request,
            cancel: Arc::new(AtomicBool::new(false)),
            running: Mutex::new(()),
        })
    }
    fn run(&self, py: Python<'_>) -> PyResult<String> {
        crate::interrupt::run(py, &self.cancel, || {
            let _guard = self.running.try_lock().map_err(|_| {
                PyRuntimeError::new_err("concurrent inference calibration is unsupported")
            })?;
            let data = beam_runtime::inference_job::run(&self.request, self.cancel.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            serde_json::to_string(&data).map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(native_depth_budget, module)?)?;
    module.add_function(wrap_pyfunction!(touch_bfs_worst_case_entries, module)?)?;
    module.add_function(wrap_pyfunction!(validate_options, module)?)?;
    module.add_function(wrap_pyfunction!(validate_search_controls, module)?)?;
    module.add_function(wrap_pyfunction!(build_graph_contract, module)?)?;
    module.add_function(wrap_pyfunction!(inference_coarse_batches, module)?)?;
    module.add_function(wrap_pyfunction!(inference_refinement_batches, module)?)?;
    module.add_function(wrap_pyfunction!(inference_probe_time_allowance, module)?)?;
    module.add_function(wrap_pyfunction!(inference_capacity_cache_reusable, module)?)?;
    module.add_function(wrap_pyfunction!(inference_select_compatible, module)?)?;
    module.add_class::<InferenceCalibration>()?;
    module.add_class::<DownstreamCalibration>()?;
    module.add_class::<FrontierCalibration>()?;
    Ok(())
}
