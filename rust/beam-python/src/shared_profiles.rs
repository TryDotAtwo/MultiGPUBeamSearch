use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::path::PathBuf;

#[pyfunction]
fn find_shared_profile(profile_root: PathBuf, identity_json: &str) -> PyResult<Option<String>> {
    let identity: serde_json::Value = serde_json::from_str(identity_json)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let Some(record) = beam_runtime::shared_profiles::find_profile(&profile_root, &identity) else {
        return Ok(None);
    };
    serde_json::to_string(&record)
        .map(Some)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(find_shared_profile, module)?)?;
    Ok(())
}
