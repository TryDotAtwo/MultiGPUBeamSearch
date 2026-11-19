//! CNOT expansion dispatch for the Rust-owned compatibility controller.
//!
//! Existing scorer calls and CUDA operators keep their current API and data
//! layout. PyTorch continues to own the parent, score, candidate, and owner
//! tensors; the Rust tensor batch path handles CPU/older-operator fallback.
use pyo3::{
    exceptions::{PyOverflowError, PyRuntimeError},
    prelude::*,
};

pub(crate) fn expand<'py>(
    py: Python<'py>,
    stages: &Bound<'py, PyAny>,
    begin: u64,
    k: u32,
) -> PyResult<Bound<'py, PyAny>> {
    let world = stages.getattr("world")?;
    let parents = stages.getattr("parents")?;
    let lex = stages.getattr("lex")?;
    let scorer = stages.getattr("scorer")?;
    let mode = stages.getattr("mode")?;
    let mode_text: String = mode.extract()?;

    let native_ops = if parents.getattr("is_cuda")?.is_truthy()? {
        Some(
            py.import("multigpubeamsearch.cnot_native")?
                .call_method0("_cuda_ops")?,
        )
    } else {
        None
    };
    let has_cuda_expand = match native_ops.as_ref() {
        Some(ops) => ops.hasattr("expand_candidates")?,
        None => false,
    };
    if !has_cuda_expand {
        let n: i64 = stages.getattr("n")?.extract()?;
        let words: i64 = stages.getattr("words")?.extract()?;
        return crate::cnot_native_batch::expand(
            py, &world, &parents, &lex, begin, &scorer, &mode_text, k, n, words, None,
        );
    }

    let ops = native_ops
        .ok_or_else(|| PyRuntimeError::new_err("CNOT expansion operators are unavailable"))?;
    let count = parents.len()?;
    let n: u64 = stages.getattr("n")?.extract()?;
    let alphabet = n
        .checked_mul(n.saturating_sub(1))
        .ok_or_else(|| PyOverflowError::new_err("CNOT move alphabet exceeds uint64"))?;
    let scores = if count != 0 && u64::from(k) < alphabet {
        let values = scorer.call_method1("parent_scores", (&parents, &mode))?;
        let expected = [
            i64::try_from(count)
                .map_err(|_| PyOverflowError::new_err("CNOT parent count exceeds int64"))?,
            i64::try_from(alphabet)
                .map_err(|_| PyOverflowError::new_err("CNOT move alphabet exceeds int64"))?,
        ];
        let device = world.getattr("device")?;
        crate::cnot_native_batch::scores(py, values, &expected, &device)?
    } else {
        py.None().into_bound(py)
    };

    let rank: u32 = world.getattr("rank")?.extract()?;
    let size: u32 = world.getattr("size")?.extract()?;
    let proposal = mode_text == "proposal";
    ops.call_method1(
        "expand_candidates",
        (&parents, &lex, scores, begin, rank, size, k, proposal),
    )
}
