//! CNOT distinct-state and scoring dispatch for the Rust-owned controller.
//!
//! Existing state identity, CUDA operators, scorer API and tensor ownership are
//! unchanged. The Rust tensor batch path handles CPU and older CUDA libraries.
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyDict, PyTuple},
};

pub(crate) fn deduplicate_score<'py>(
    py: Python<'py>,
    stages: &Bound<'py, PyAny>,
    strict_backend: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let world = stages.getattr("world")?;
    let incoming = stages.getattr("incoming")?;
    let closed = stages.getattr("closed")?;
    let scorer = stages.getattr("scorer")?;
    let n = stages.getattr("n")?;
    let words = stages.getattr("words")?;

    let native_ops = if incoming.getattr("is_cuda")?.is_truthy()? {
        Some(
            py.import("multigpubeamsearch.cnot_native")?
                .call_method0("_cuda_ops")?,
        )
    } else {
        None
    };
    let has_native_distinct = match native_ops.as_ref() {
        Some(ops) => ops.hasattr("distinct_candidates")?,
        None => false,
    };
    if !has_native_distinct {
        return crate::cnot_native_batch::distinct(
            py,
            &world,
            &incoming,
            &closed,
            n.extract()?,
            words.extract()?,
            &scorer,
            strict_backend,
            None,
        );
    }

    let ops = native_ops
        .ok_or_else(|| PyRuntimeError::new_err("CNOT distinct-state operators are unavailable"))?;
    let distinct = ops.call_method1("distinct_candidates", (&incoming, &closed, &n, &words))?;
    let distinct = distinct.downcast::<PyTuple>()?;
    if distinct.len() != 2 {
        return Err(PyRuntimeError::new_err(
            "CNOT distinct-state operator returned an invalid adapter result",
        ));
    }
    let children = distinct.get_item(0)?;
    let meta = distinct.get_item(1)?;
    let child_count = children.len()?;
    let values = if child_count != 0 {
        let raw = crate::cnot_scorer_dispatch::child_values(
            py, &world, &scorer, &children, strict_backend,
        )?;
        let expected = [i64::try_from(child_count).map_err(|_| {
            pyo3::exceptions::PyOverflowError::new_err("CNOT distinct count exceeds int64")
        })?];
        let device = world.getattr("device")?;
        crate::cnot_native_batch::scores(py, raw, &expected, &device)?
    } else {
        let torch = py.import("torch")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", torch.getattr("float64")?)?;
        kwargs.set_item("device", world.getattr("device")?)?;
        torch.call_method("empty", (0,), Some(&kwargs))?
    };
    PyTuple::new(py, [children, meta, values]).map(Bound::into_any)
}
