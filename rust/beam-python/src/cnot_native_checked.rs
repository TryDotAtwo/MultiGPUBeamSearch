//! Rank-wide error coordination for fallible CNOT stages.
//!
//! The operation still runs on the calling thread. Only the existing world
//! error vote and, on failure, object gather are used; Torch owns its buffers.
use pyo3::{
    exceptions::{PyException, PyRuntimeError},
    prelude::*,
    types::PyList,
};

pub(crate) fn checked<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    stage: &str,
    result: PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let size: usize = world.getattr("size")?.extract()?;
    if size == 1 {
        return result;
    }
    let (value, error) = match result {
        Ok(value) => (Some(value), None),
        Err(error) => {
            if !error.is_instance_of::<PyException>(py) {
                return Err(error);
            }
            let rank: usize = world.getattr("rank")?.extract()?;
            let name: String = error.get_type(py).getattr("__name__")?.extract()?;
            let message = error.value(py).str()?.to_str()?.to_owned();
            (None, Some(format!("rank {rank}: {name}: {message}")))
        }
    };
    let vote = world.call_method1("longs", (vec![i64::from(error.is_some())],))?;
    if world
        .call_method1("sum", (vote,))?
        .call_method0("item")?
        .is_truthy()?
    {
        let errors = PyList::new(py, (0..size).map(|_| py.None()))?;
        py.import("torch.distributed")?
            .call_method1("all_gather_object", (&errors, error))?;
        let messages: Vec<Option<String>> = errors.extract()?;
        return Err(PyRuntimeError::new_err(format!(
            "{stage} failed: {}",
            messages
                .into_iter()
                .flatten()
                .filter(|message| !message.is_empty())
                .collect::<Vec<_>>()
                .join(" | ")
        )));
    }
    Ok(value.unwrap_or_else(|| py.None().into_bound(py)))
}

#[pyfunction]
fn run_cnot_native_checked<'py>(
    py: Python<'py>,
    world: Bound<'py, PyAny>,
    stage: &str,
    callback: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    checked(py, &world, stage, callback.call0())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_cnot_native_checked, module)?)
}
