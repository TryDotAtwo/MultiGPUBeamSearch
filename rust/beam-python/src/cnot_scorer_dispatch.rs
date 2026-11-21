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
) -> PyResult<Bound<'py, PyAny>> {
    let rust_tqp = scorer.is_instance_of::<PyTqpScorer>();
    if strict_acceptance() && !rust_tqp {
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
    if strict_acceptance() && rows.len()? > 0 {
        trace_native_tqp(world, scorer, "parent_scores", rows.len()?)?;
    }
    Ok(result)
}

pub(crate) fn child_values<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    scorer: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let rust_tqp = scorer.is_instance_of::<PyTqpScorer>();
    if strict_acceptance() && !rust_tqp {
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
    if strict_acceptance() && rows.len()? > 0 {
        trace_native_tqp(world, scorer, "child_values", rows.len()?)?;
    }
    Ok(result)
}

fn strict_acceptance() -> bool {
    matches!(
        std::env::var("MGBS_CNOT_STRICT_BACKEND").as_deref(),
        Ok("1")
    )
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
    eprintln!(
        "CNOT_RUST_TQP_SCORER_DISPATCH rank={rank} world={size} pid={pid} device={device} implementation=RustTqpScorer abi=1 scorer_abi=1 operation={operation} rows={rows} checkpoint_sha256={checkpoint} search_manifest_sha256={manifest} scorer_library_sha256={scorer_library}"
    );
    Ok(())
}
