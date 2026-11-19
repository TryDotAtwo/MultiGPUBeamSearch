//! CNOT rank-local batch orchestration using the existing tensor operators.
//!
//! Torch owns tensor storage and executes the same operations as the Python
//! reference. The CUDA operators, ordering, and score precision are unchanged.
use crate::cnot_scorer_dispatch;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyTuple},
};

fn slice<'py>(
    py: Python<'py>,
    begin: Option<i64>,
    end: Option<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("builtins")?.getattr("slice")?.call1((begin, end))
}

fn pack_route<'py>(
    py: Python<'py>,
    source_rank: u32,
    moves: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if source_rank > u32::from(u16::MAX) {
        return Err(PyValueError::new_err(
            "CNOT source rank must fit uint16",
        ));
    }
    let torch = py.import("torch")?;
    let dtype = moves.getattr("dtype")?;
    let int64 = torch.getattr("int64")?;
    let uint16 = torch.getattr("uint16")?;
    let is_int64 = dtype.call_method1("__eq__", (&int64,))?.is_truthy()?;
    let is_uint16 = dtype.call_method1("__eq__", (&uint16,))?.is_truthy()?;
    if !is_int64 && !is_uint16 {
        return Err(PyValueError::new_err("CNOT moves must be uint16 or int64"));
    }
    if is_int64 {
        let nonnegative = moves.call_method1("__ge__", (0,))?;
        let below_limit = moves.call_method1("__le__", (i64::from(u16::MAX),))?;
        let valid = torch
            .call_method1("logical_and", (nonnegative, below_limit))?
            .call_method0("all")?;
        torch.call_method1(
            "_assert_async",
            (valid, "CNOT move must fit uint16"),
        )?;
    }
    let moves = moves.call_method1("to", (&int64,))?;
    moves.call_method1("__or__", (i64::from(source_rank) << 16,))
}
pub(crate) fn scores<'py>(
    py: Python<'py>,
    tensor: Bound<'py, PyAny>,
    expected: &[i64],
    device: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = PyTuple::new(py, expected.iter().copied())?;
    if tensor
        .getattr("shape")?
        .call_method1("__ne__", (&shape,))?
        .is_truthy()?
        || tensor
            .getattr("device")?
            .call_method1("__ne__", (device,))?
            .is_truthy()?
        || !tensor.call_method0("is_floating_point")?.is_truthy()?
    {
        return Err(PyValueError::new_err(format!(
            "expected floating scores on {} with shape {}",
            device.str()?,
            shape.str()?
        )));
    }
    if !py
        .import("torch")?
        .call_method1("isfinite", (&tensor,))?
        .call_method0("all")?
        .call_method0("item")?
        .is_truthy()?
    {
        return Err(PyValueError::new_err("nonfinite CNOT model score"));
    }
    Ok(tensor)
}

pub(crate) fn expand<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    parents: &Bound<'py, PyAny>,
    lex: &Bound<'py, PyAny>,
    begin: u64,
    scorer: &Bound<'py, PyAny>,
    mode: &str,
    k: u32,
    n: i64,
    words: i64,
    mut device_domain: Option<&mut beam_native_sys::cnot_device_bridge::DeviceDomain>,
) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import("torch")?;
    let native = py.import("multigpubeamsearch.cnot_native")?;
    let device = world.getattr("device")?;
    let count = parents.len()?;
    let alphabet = n * (n - 1);
    let arange_kwargs = PyDict::new(py);
    arange_kwargs.set_item("device", &device)?;
    let all_ids = torch.call_method("arange", (alphabet,), Some(&arange_kwargs))?;
    let moves = if count > 0 && i64::from(k) < alphabet {
        let raw = cnot_scorer_dispatch::parent_scores(py, scorer, parents, mode)?;
        let valid = scores(py, raw, &[count as i64, alphabet], &device)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dim", 1)?;
        kwargs.set_item("descending", mode == "proposal")?;
        kwargs.set_item("stable", true)?;
        torch
            .call_method("argsort", (valid,), Some(&kwargs))?
            .get_item((slice(py, None, None)?, slice(py, None, Some(i64::from(k)))?))?
    } else {
        all_ids
            .get_item(slice(py, None, Some(i64::from(k)))?)?
            .call_method1("expand", (count, k))?
    };
    let indices = torch
        .call_method("arange", (count,), Some(&arange_kwargs))?
        .call_method1("repeat_interleave", (k,))?;
    let moves = moves.call_method1("reshape", (-1,))?;
    let children = match device_domain.as_deref_mut() {
        Some(domain) => domain.children(py, parents, &indices, &moves)?,
        None => native.call_method1("cnot_children", (parents, &indices, &moves))?,
    };
    let parent_slots = indices.call_method1("__add__", (begin,))?;
    let generation = parent_slots
        .call_method1("__mul__", (alphabet,))?
        .call_method1(
            "__add__",
            (torch
                .call_method("arange", (k,), Some(&arange_kwargs))?
                .call_method1("repeat", (count,))?,),
        )?;
    let path_keys = lex
        .get_item(&indices)?
        .call_method1("__mul__", (alphabet,))?
        .call_method1("__add__", (&moves,))?;
    let source_rank: u32 = world.getattr("rank")?.extract()?;
    let route = pack_route(py, source_rank, &moves)?;
    let meta = torch.call_method1(
        "stack",
        (
            PyTuple::new(py, [parent_slots, path_keys, generation, route])?,
            1,
        ),
    )?;
    let rows = children.call_method1("reshape", (-1, n * words))?;
    let candidates = torch.call_method1("cat", (PyTuple::new(py, [meta, rows])?, 1))?;
    let owners = match device_domain.as_deref_mut() {
        Some(domain) => domain.owners(py, &children)?,
        None => native.call_method1("cnot_owners", (&children, world.getattr("size")?))?,
    };
    Ok(PyTuple::new(py, [candidates, owners])?.into_any())
}

pub(crate) fn distinct<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    incoming: &Bound<'py, PyAny>,
    closed: &Bound<'py, PyAny>,
    n: i64,
    words: i64,
    scorer: &Bound<'py, PyAny>,
    device_domain: Option<&mut beam_native_sys::cnot_device_bridge::DeviceDomain>,
) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import("torch")?;
    let native = py.import("multigpubeamsearch.cnot_native")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("stable", true)?;
    let generations = incoming.get_item((slice(py, None, None)?, 2))?;
    let order = torch.call_method("argsort", (generations,), Some(&kwargs))?;
    let incoming = incoming.get_item(order)?;
    let children = incoming
        .get_item((slice(py, None, None)?, slice(py, Some(4), None)?))?
        .call_method1("reshape", (-1, n, words))?
        .call_method0("contiguous")?;
    let first = match device_domain {
        Some(domain) => domain.first_unseen(py, closed, &children)?,
        None => native.call_method1("cnot_first_unseen", (closed, &children))?,
    };
    let children = children.get_item(&first)?;
    let meta = incoming.get_item((&first, slice(py, None, Some(4))?))?;
    let device = world.getattr("device")?;
    let values = if children.len()? > 0 {
        let raw = cnot_scorer_dispatch::child_values(py, scorer, &children)?;
        scores(py, raw, &[children.len()? as i64], &device)?
    } else {
        let empty_kwargs = PyDict::new(py);
        empty_kwargs.set_item("dtype", torch.getattr("float64")?)?;
        empty_kwargs.set_item("device", device)?;
        torch.call_method("empty", (0,), Some(&empty_kwargs))?
    };
    Ok(PyTuple::new(py, [children, meta, values])?.into_any())
}
