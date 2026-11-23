//! Dispatch CNOT scorer packets without re-entering Python for the built-in
//! Rust checkpoint scorer. Third-party Python scorers keep the existing
//! method protocol.
use crate::cnot_tqp_scorer::PyTqpScorer;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub(crate) fn parent_scores<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    scorer: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    mode: &str,
    strict: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let rust_tqp = scorer.is_instance_of::<PyTqpScorer>();
    if strict && !rust_tqp {
        return Err(PyRuntimeError::new_err(
            "strict CNOT acceptance observed a non-Rust parent scorer",
        ));
    }
    let result = if rust_tqp {
        scorer
            .extract::<PyRef<'_, PyTqpScorer>>()?
            .parent_scores_native(py, rows.clone(), mode)?
    } else {
        scorer.call_method1("parent_scores", (rows, mode))?
    };
    if strict && rows.len()? > 0 {
        trace_native_tqp(world, scorer, "parent_scores", rows.len()?)?;
    }
    Ok(result)
}

pub(crate) fn child_values<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    scorer: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    strict: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let rust_tqp = scorer.is_instance_of::<PyTqpScorer>();
    if strict && !rust_tqp {
        return Err(PyRuntimeError::new_err(
            "strict CNOT acceptance observed a non-Rust child scorer",
        ));
    }
    let result = if rust_tqp {
        scorer
            .extract::<PyRef<'_, PyTqpScorer>>()?
            .child_values_native(py, rows.clone())?
    } else {
        scorer.call_method1("child_values", (rows,))?
    };
    if strict && rows.len()? > 0 {
        trace_native_tqp(world, scorer, "child_values", rows.len()?)?;
    }
    Ok(result)
}

fn trace_native_tqp(
    world: &Bound<'_, PyAny>,
    scorer: &Bound<'_, PyAny>,
    operation: &str,
    rows: usize,
) -> PyResult<()> {
    let rank: u32 = world.getattr("rank")?.extract()?;
    let size: u32 = world.getattr("size")?.extract()?;
    let pid = std::process::id();
    let device = scorer.getattr("device")?.str()?.to_str()?.to_owned();
    let metadata = scorer.getattr("metadata")?;
    let checkpoint: String = metadata.get_item("checkpoint_sha256")?.extract()?;
    let manifest: String = metadata.get_item("search_manifest_sha256")?.extract()?;
    let scorer_library: String = metadata.get_item("scorer_library_sha256")?.extract()?;
    let fields = [
        ("implementation".to_owned(), "RustTqpScorer".to_owned()),
        ("abi".to_owned(), "1".to_owned()),
        ("scorer_abi".to_owned(), "1".to_owned()),
        ("operation".to_owned(), operation.to_owned()),
        ("rows".to_owned(), rows.to_string()),
        ("checkpoint_sha256".to_owned(), checkpoint),
        ("search_manifest_sha256".to_owned(), manifest),
        ("scorer_library_sha256".to_owned(), scorer_library),
    ]
    .into_iter()
    .collect();
    let py = world.py();
    let active_device = world.getattr("device")?;
    let observation = crate::cnot_native_stages::observe_acceptance_gpu(py, world, &active_device)?;
    crate::cnot_acceptance::emit(
        "tqp_scorer_dispatch",
        rank,
        size,
        pid,
        &device,
        &fields,
        observation.as_ref(),
    )
}
