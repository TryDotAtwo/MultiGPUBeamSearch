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

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(resolve_inference_backend, module)?)?;
    Ok(())
}
