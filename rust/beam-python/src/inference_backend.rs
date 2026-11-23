use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

#[pyfunction]
fn resolve_inference_backend(request_json: &str) -> PyResult<String> {
    let request: beam_runtime::inference_backend::Request =
        serde_json::from_str(request_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    beam_runtime::inference_backend::resolve(&request)
        .map(|backend| backend.as_str().to_owned())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction]
fn resolve_inference_plan(request_json: &str) -> PyResult<String> {
    let request: beam_runtime::inference_backend::Request =
        serde_json::from_str(request_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let plan = beam_runtime::inference_backend::resolve_plan(&request)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    serde_json::to_string(&serde_json::json!({
        "backend": plan.backend.as_str(),
        "policy": plan.policy,
        "inference_row_limit": plan.inference_row_limit,
    }))
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction]
fn resolve_microbatch_plan(request_json: &str) -> PyResult<String> {
    let request: beam_runtime::microbatch::Request = serde_json::from_str(request_json)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let plan = beam_runtime::microbatch::resolve(&request)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    serde_json::to_string(&plan)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(resolve_inference_backend, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_inference_plan, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_microbatch_plan, module)?)?;
    Ok(())
}
