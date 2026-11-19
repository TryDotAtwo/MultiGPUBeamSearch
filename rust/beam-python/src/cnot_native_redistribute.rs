//! CNOT winner routing, history rows and audit collection for compatibility stages.
//!
//! Rust owns the stage orchestration. Torch tensors and the existing CUDA
//! operators own all state and device memory; the established process-group
//! exchange and collective order are preserved.
use pyo3::{
    exceptions::{PyOverflowError, PyRuntimeError},
    prelude::*,
    types::{PyDict, PyTuple},
};

pub(crate) fn redistribute<'py>(
    py: Python<'py>,
    stages: &Bound<'py, PyAny>,
    depth: u64,
) -> PyResult<(u64, Py<PyAny>, Option<Py<PyAny>>)> {
    let world = stages.getattr("world")?;
    let ranks = stages.getattr("ranks")?;
    let local = stages.getattr("local")?;
    let new_lex = stages.getattr("new_lex")?;
    let meta = stages.getattr("meta")?;
    let children = stages.getattr("children")?;
    let closed = stages.getattr("closed")?;
    let count = local.len()?;
    let n: usize = stages.getattr("n")?.extract()?;
    let words: usize = stages.getattr("words")?.extract()?;
    let columns = n
        .checked_mul(words)
        .ok_or_else(|| PyOverflowError::new_err("CNOT state row width exceeds usize"))?;
    let rank: u32 = world.getattr("rank")?.extract()?;
    let world_size: u32 = world.getattr("size")?.extract()?;

    let ops = if children.getattr("is_cuda")?.is_truthy()? {
        Some(
            py.import("multigpubeamsearch.cnot_native")?
                .call_method0("_cuda_ops")?,
        )
    } else {
        None
    };
    let use_native = if let Some(ops) = ops.as_ref() {
        !ops.is_none()
            && ops.hasattr("winner_packet")?
            && ops.hasattr("sort_received")?
            && ops.hasattr("frontier_views")?
    } else {
        false
    };

    let (new_closed, wire, targets) = if use_native {
        let ops = ops
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CNOT winner operators unavailable"))?;
        let packet = ops.call_method1(
            "winner_packet",
            (
                &ranks, &local, &new_lex, &meta, &children, &closed, rank, world_size,
            ),
        )?;
        (packet.get_item(0)?, packet.get_item(1)?, packet.get_item(2)?)
    } else {
        let torch = py.import("torch")?;
        let slots = ranks
            .call_method1("__eq__", (rank,))?
            .call_method0("nonzero")?
            .call_method0("flatten")?;
        let kept = local.get_item(&slots)?;
        let owned = children.get_item(&kept)?;
        let new_closed = torch.call_method1(
            "cat",
            (PyTuple::new(py, [closed.clone(), owned.clone()])?,),
        )?;
        let records = torch.call_method1(
            "stack",
            (
                PyTuple::new(
                    py,
                    [
                        slots.clone(),
                        new_lex.get_item(&slots)?,
                        meta.get_item((&kept, 0))?,
                        meta.get_item((&kept, 3))?,
                        ranks.get_item(&slots)?,
                    ],
                )?,
                1,
            ),
        )?;
        let rows = owned.call_method1("reshape", (kept.len()?, columns))?;
        let wire = torch.call_method1("cat", (PyTuple::new(py, [records, rows])?, 1))?;
        let targets = slots
            .call_method1("__mul__", (world_size,))?
            .call_method1("__floordiv__", (count.max(1),))?;
        (new_closed, wire, targets)
    };

    // Preserve the existing mutation order: closed rows update before the
    // collective exchange begins.
    stages.setattr("closed", new_closed)?;
    let received =
        crate::cnot_native_exchange::exchange(py, &world, &wire, &targets, true)?;
    let received = if use_native {
        ops.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CNOT winner operators unavailable"))?
            .call_method1("sort_received", (received,))?
    } else {
        let order = py
            .import("torch")?
            .call_method1("argsort", (received.call_method1("select", (1, 0))?,))?;
        received.get_item(order)?
    };

    let parents = stages.getattr("parents")?;
    let candidates = stages.getattr("candidates")?;
    let owners = stages.getattr("owners")?;
    let remote: i64 = owners
        .call_method1("__ne__", (rank,))?
        .call_method0("sum")?
        .call_method0("item")?
        .extract()?;
    let stat_values = vec![
        i64::try_from(parents.len()?)
            .map_err(|_| PyOverflowError::new_err("CNOT parent count exceeds int64"))?,
        i64::try_from(candidates.len()?)
            .map_err(|_| PyOverflowError::new_err("CNOT candidate count exceeds int64"))?,
        remote,
        i64::try_from(children.len()?)
            .map_err(|_| PyOverflowError::new_err("CNOT distinct count exceeds int64"))?,
        i64::try_from(received.len()?)
            .map_err(|_| PyOverflowError::new_err("CNOT winner count exceeds int64"))?,
    ];
    let stats = world
        .call_method1("gather_fixed", (world.call_method1("longs", (stat_values,))?,))?
        .call_method0("tolist")?;
    let rows: Vec<Vec<i64>> = stats.extract()?;
    let distinct = rows.iter().try_fold(0_i64, |sum, row| {
        let count = row
            .get(3)
            .ok_or_else(|| PyRuntimeError::new_err("invalid CNOT rank statistics"))?;
        sum.checked_add(*count)
            .ok_or_else(|| PyOverflowError::new_err("CNOT rank statistics overflow"))
    })?;

    let layer = PyDict::new(py);
    layer.set_item("depth", depth)?;
    layer.set_item("by_rank", stats)?;
    layer.set_item("distinct", distinct)?;
    layer.set_item(
        "retained",
        u64::try_from(count)
            .map_err(|_| PyOverflowError::new_err("CNOT retained count exceeds uint64"))?,
    )?;
    if stages.getattr("audit")?.is_truthy()? {
        let torch = py.import("torch")?;
        let values = stages
            .getattr("values")?
            .call_method1("to", (torch.getattr("float64")?,))?
            .call_method0("contiguous")?
            .call_method1("view", (torch.getattr("int64")?,))?
            .call_method1("unsqueeze", (1,))?;
        let flat = children.call_method1("reshape", (children.len()?, columns))?;
        let payload =
            torch.call_method1("cat", (PyTuple::new(py, [values, meta, flat])?, 1))?;
        let candidate_rows = world
            .call_method1("gather_rows", (payload,))?
            .call_method0("cpu")?
            .call_method0("tolist")?;
        let winner_rows = world
            .call_method1("gather_rows", (&received,))?
            .call_method0("cpu")?
            .call_method0("tolist")?;
        layer.set_item("candidates", candidate_rows)?;
        layer.set_item("winners", winner_rows)?;
    }

    if count == 0 {
        return Ok((0, layer.into_any().unbind(), None));
    }

    let (history, next_parents, lex) = if use_native {
        let parts = ops
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CNOT winner operators unavailable"))?
            .call_method1("frontier_views", (&received, n, words))?;
        (parts.get_item(0)?, parts.get_item(1)?, parts.get_item(2)?)
    } else {
        let history = received
            .call_method1("narrow", (1, 2, 3))?
            .call_method0("contiguous")?;
        let next_parents = received
            .call_method1("narrow", (1, 5, columns))?
            .call_method1("reshape", (-1, n, words))?
            .call_method0("contiguous")?;
        let lex = received
            .call_method1("select", (1, 1))?
            .call_method0("contiguous")?;
        (history, next_parents, lex)
    };
    stages.setattr("parents", next_parents)?;
    stages.setattr("lex", lex)?;
    Ok((
        u64::try_from(count)
            .map_err(|_| PyOverflowError::new_err("CNOT retained count exceeds uint64"))?,
        layer.into_any().unbind(),
        Some(history.unbind()),
    ))
}
