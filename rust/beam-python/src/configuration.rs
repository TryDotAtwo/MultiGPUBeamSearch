use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
#[pyfunction]
fn validate_options(options: &str) -> PyResult<String> {
    let options: beam_core::config::NativeOptions =
        serde_json::from_str(options).map_err(|e| PyValueError::new_err(e.to_string()))?;
    options
        .validate()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&options).map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyclass]
struct InferenceCalibration {
    request: beam_runtime::inference_job::Request,
    cancel: Arc<AtomicBool>,
    running: Mutex<()>,
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
    module.add_function(wrap_pyfunction!(validate_options, module)?)?;
    module.add_class::<InferenceCalibration>()?;
    Ok(())
}
