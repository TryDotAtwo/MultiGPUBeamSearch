//! Dispatch CNOT scorer packets without re-entering Python for the built-in
//! Rust checkpoint scorer. Third-party Python scorers keep the existing
//! method protocol.
use crate::cnot_tqp_scorer::PyTqpScorer;
use pyo3::prelude::*;

pub(crate) fn parent_scores<'py>(
    py: Python<'py>,
    scorer: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    mode: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if scorer.is_instance_of::<PyTqpScorer>() {
        return scorer
            .extract::<PyRef<'_, PyTqpScorer>>()?
            .parent_scores_native(py, rows.clone(), mode);
    }
    scorer.call_method1("parent_scores", (rows, mode))
}

pub(crate) fn child_values<'py>(
    py: Python<'py>,
    scorer: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if scorer.is_instance_of::<PyTqpScorer>() {
        return scorer
            .extract::<PyRef<'_, PyTqpScorer>>()?
            .child_values_native(py, rows.clone());
    }
    scorer.call_method1("child_values", (rows,))
}
