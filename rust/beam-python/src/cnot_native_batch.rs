//! CNOT rank-local batch orchestration using the existing tensor operators.
//!
//! Torch owns tensor storage and executes the same operations as the Python
//! reference. The CUDA operators, ordering, and score precision are unchanged.
use crate::cnot_scorer_dispatch;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList, PyTuple},
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
        return Err(PyValueError::new_err("CNOT source rank must fit uint16"));
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
        torch.call_method1("_assert_async", (valid, "CNOT move must fit uint16"))?;
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

fn cpu_children<'py>(
    py: Python<'py>,
    parents: &Bound<'py, PyAny>,
    parent_indices: &Bound<'py, PyAny>,
    moves: &Bound<'py, PyAny>,
    n: i64,
    words: i64,
) -> PyResult<Bound<'py, PyAny>> {
    if parents.getattr("is_cuda")?.is_truthy()? || !(2..=256).contains(&n) {
        return Err(PyValueError::new_err("invalid CPU CNOT parent tensor"));
    }
    let expected_words = (n + 63) / 64;
    if words != expected_words {
        return Err(PyValueError::new_err("invalid CNOT packed row width"));
    }
    let parent_shape: Vec<i64> = parents.getattr("shape")?.extract()?;
    let index_shape: Vec<i64> = parent_indices.getattr("shape")?.extract()?;
    let move_shape: Vec<i64> = moves.getattr("shape")?.extract()?;
    if parent_shape.len() != 3 || parent_shape[1] != n || parent_shape[2] != words {
        return Err(PyValueError::new_err("invalid CNOT parent tensor shape"));
    }
    if index_shape.len() != 1 || move_shape != index_shape {
        return Err(PyValueError::new_err(
            "expected equally sized one-dimensional CNOT descriptors",
        ));
    }

    let torch = py.import("torch")?;
    let int64 = torch.getattr("int64")?;
    let uint16 = torch.getattr("uint16")?;
    let parents_are_int64 = parents
        .getattr("dtype")?
        .call_method1("__eq__", (&int64,))?
        .is_truthy()?;
    let indices_are_int64 = parent_indices
        .getattr("dtype")?
        .call_method1("__eq__", (&int64,))?
        .is_truthy()?;
    let moves_are_int64 = moves
        .getattr("dtype")?
        .call_method1("__eq__", (&int64,))?
        .is_truthy()?;
    let moves_are_uint16 = moves
        .getattr("dtype")?
        .call_method1("__eq__", (&uint16,))?
        .is_truthy()?;
    if !parents_are_int64 || !indices_are_int64 || !(moves_are_int64 || moves_are_uint16) {
        return Err(PyValueError::new_err(
            "expected int64 CNOT states and indices, and int64/uint16 moves",
        ));
    }
    let parent_device = parents.getattr("device")?;
    for descriptor in [parent_indices, moves] {
        if descriptor
            .getattr("device")?
            .call_method1("__ne__", (&parent_device,))?
            .is_truthy()?
        {
            return Err(PyValueError::new_err(
                "CNOT parents and descriptors must use the same device",
            ));
        }
    }

    let n_usize =
        usize::try_from(n).map_err(|_| PyValueError::new_err("CNOT wire count overflow"))?;
    let words_usize =
        usize::try_from(words).map_err(|_| PyValueError::new_err("CNOT row width overflow"))?;
    let matrix_words = n_usize
        .checked_mul(words_usize)
        .ok_or_else(|| PyValueError::new_err("CNOT matrix size overflow"))?;
    let parent_count = usize::try_from(parent_shape[0])
        .map_err(|_| PyValueError::new_err("CNOT parent count overflow"))?;
    let expected_parent_words = parent_count
        .checked_mul(matrix_words)
        .ok_or_else(|| PyValueError::new_err("CNOT parent tensor size overflow"))?;
    let parents_flat = parents
        .call_method0("contiguous")?
        .call_method1("reshape", (-1,))?
        .call_method0("tolist")?
        .extract::<Vec<i64>>()?;
    if parents_flat.len() != expected_parent_words {
        return Err(PyValueError::new_err("invalid CNOT parent tensor storage"));
    }
    let indices: Vec<i64> = parent_indices.call_method0("tolist")?.extract()?;
    let moves_i64: Vec<i64> = moves
        .call_method1("to", (&int64,))?
        .call_method0("tolist")?
        .extract()?;
    if indices.len() != moves_i64.len() {
        return Err(PyValueError::new_err("CNOT descriptor length mismatch"));
    }

    let output_words = indices
        .len()
        .checked_mul(matrix_words)
        .ok_or_else(|| PyValueError::new_err("CNOT output size overflow"))?;
    let mut output = vec![0u64; output_words];
    for (child, (&parent_index, &move_id)) in indices.iter().zip(&moves_i64).enumerate() {
        let parent_index = usize::try_from(parent_index)
            .map_err(|_| PyValueError::new_err("CNOT parent index is outside the batch"))?;
        if parent_index >= parent_count {
            return Err(PyValueError::new_err(
                "CNOT parent index is outside the batch",
            ));
        }
        let move_id = u32::try_from(move_id)
            .map_err(|_| PyValueError::new_err("CNOT move is outside the physical alphabet"))?;
        let parent_start = parent_index
            .checked_mul(matrix_words)
            .ok_or_else(|| PyValueError::new_err("CNOT parent offset overflow"))?;
        let child_start = child
            .checked_mul(matrix_words)
            .ok_or_else(|| PyValueError::new_err("CNOT output offset overflow"))?;
        for word in 0..matrix_words {
            output[child_start + word] = parents_flat[parent_start + word] as u64;
        }
        beam_core::cnot::apply_gate_words(
            n as u32,
            &mut output[child_start..child_start + matrix_words],
            move_id,
        )
        .map_err(|_| PyValueError::new_err("CNOT move or packed matrix is invalid"))?;
    }

    let values = PyList::new(py, output.iter().copied().map(|word| word as i64))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", int64)?;
    kwargs.set_item("device", parent_device)?;
    let tensor = torch.call_method("tensor", (values,), Some(&kwargs))?;
    tensor.call_method1("reshape", (indices.len() as i64, n, words))
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
    let children = if let Some(domain) = device_domain.as_deref_mut() {
        domain.children(py, parents, &indices, &moves)?
    } else if parents.getattr("is_cuda")?.is_truthy()? {
        native.call_method1("cnot_children", (parents, &indices, &moves))?
    } else {
        cpu_children(py, parents, &indices, &moves, n, words)?
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

fn cpu_first_unseen<'py>(
    py: Python<'py>,
    closed: &Bound<'py, PyAny>,
    children: &Bound<'py, PyAny>,
    n: i64,
    words: i64,
) -> PyResult<Bound<'py, PyAny>> {
    if !(1..=256).contains(&n)
        || words != (n + 63) / 64
        || closed.getattr("is_cuda")?.is_truthy()?
        || children.getattr("is_cuda")?.is_truthy()?
    {
        return Err(PyValueError::new_err("invalid CPU CNOT state tensors"));
    }
    let closed_shape: Vec<i64> = closed.getattr("shape")?.extract()?;
    let children_shape: Vec<i64> = children.getattr("shape")?.extract()?;
    if closed_shape.len() != 3
        || children_shape.len() != 3
        || closed_shape[1] != n
        || children_shape[1] != n
        || closed_shape[2] != words
        || children_shape[2] != words
    {
        return Err(PyValueError::new_err("invalid packed CNOT matrix shape"));
    }

    let torch = py.import("torch")?;
    let int64 = torch.getattr("int64")?;
    for tensor in [closed, children] {
        if !tensor
            .getattr("dtype")?
            .call_method1("__eq__", (&int64,))?
            .is_truthy()?
            || tensor
                .getattr("device")?
                .getattr("type")?
                .extract::<String>()?
                != "cpu"
        {
            return Err(PyValueError::new_err(
                "CNOT matrices require int64 CPU tensors for Rust reference deduplication",
            ));
        }
    }
    let closed_device = closed.getattr("device")?;
    let children_device = children.getattr("device")?;
    if closed_device
        .call_method1("__ne__", (&children_device,))?
        .is_truthy()?
    {
        return Err(PyValueError::new_err(
            "closed and child CNOT matrices must share n and device",
        ));
    }

    let closed_flat: Vec<i64> = closed
        .call_method0("contiguous")?
        .call_method1("reshape", (-1,))?
        .call_method0("tolist")?
        .extract()?;
    let children_flat: Vec<i64> = children
        .call_method0("contiguous")?
        .call_method1("reshape", (-1,))?
        .call_method0("tolist")?
        .extract()?;
    let closed_words: Vec<u64> = closed_flat.into_iter().map(|word| word as u64).collect();
    let child_words: Vec<u64> = children_flat.into_iter().map(|word| word as u64).collect();
    let selected = beam_core::cnot::first_unseen_indices(n as u32, &closed_words, &child_words)
        .map_err(|_| PyValueError::new_err("invalid packed CNOT state batch"))?;

    let indices = PyList::new(py, selected.iter().map(|&index| index as i64))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", int64)?;
    kwargs.set_item("device", children_device)?;
    torch.call_method("tensor", (indices,), Some(&kwargs))
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
    let first = if let Some(domain) = device_domain {
        domain.first_unseen(py, closed, &children)?
    } else if children.getattr("is_cuda")?.is_truthy()? {
        native.call_method1("cnot_first_unseen", (closed, &children))?
    } else {
        cpu_first_unseen(py, closed, &children, n, words)?
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
