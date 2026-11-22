use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{atomic::AtomicBool, Arc},
};

#[pyfunction]
fn resolve_cluster_plan(payload: String) -> PyResult<String> {
    let request: beam_runtime::cluster::PlanRequest =
        serde_json::from_str(&payload).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let plan = beam_runtime::cluster::plan_cluster(&request).map_err(|error| match error {
        beam_runtime::cluster::ClusterError::Invalid(message) => PyValueError::new_err(message),
        beam_runtime::cluster::ClusterError::Runtime(message) => PyRuntimeError::new_err(message),
        beam_runtime::cluster::ClusterError::Io(error) => PyErr::from(error),
        beam_runtime::cluster::ClusterError::Json(error) => {
            PyValueError::new_err(error.to_string())
        }
    })?;
    serde_json::to_string(&plan).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction]
fn validate_cluster_rank(environment: BTreeMap<String, String>) -> PyResult<()> {
    beam_runtime::cluster::scheduler_rank(&environment)
        .map(|_| ())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn resolve_scheduler_rank(environment: BTreeMap<String, String>) -> PyResult<String> {
    let rank = beam_runtime::cluster::scheduler_rank(&environment)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    serde_json::to_string(&rank).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction]
fn run_cluster_rank(
    py: Python<'_>,
    command: Vec<String>,
    executable: Option<PathBuf>,
    shared_dir: PathBuf,
    run_id: Option<String>,
    timeout: f64,
    environment: BTreeMap<String, String>,
) -> PyResult<PathBuf> {
    let cancel = Arc::new(AtomicBool::new(false));
    let request = beam_runtime::cluster::Request {
        command,
        executable,
        shared_dir,
        run_id,
        timeout,
        environment,
    };
    crate::interrupt::run(py, &cancel, || {
        beam_runtime::cluster::run(request, &cancel).map_err(|error| match error {
            beam_runtime::cluster::ClusterError::Invalid(message) => PyValueError::new_err(message),
            beam_runtime::cluster::ClusterError::Io(error) => PyErr::from(error),
            other => PyRuntimeError::new_err(other.to_string()),
        })
    })
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("CLUSTER_RANK_VERSION", 1)?;
    module.add_function(wrap_pyfunction!(run_cluster_rank, module)?)?;
    module.add_function(wrap_pyfunction!(validate_cluster_rank, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_scheduler_rank, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_cluster_plan, module)?)?;
    Ok(())
}
