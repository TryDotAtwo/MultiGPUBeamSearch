//! Rank-wide CNOT goal coordination using the existing Torch/CUDA operations.
//!
//! Torch owns every tensor. This module only retains handles during one
//! coarse stage, and uses the same goal and error collectives as the reference.
use crate::cnot_native_checked::gather_error_messages_v2;
use pyo3::{
    exceptions::{PyException, PyKeyboardInterrupt, PyOverflowError, PyRuntimeError, PyZeroDivisionError},
    prelude::*,
    types::PyDict,
};

fn local_goal<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    parents: &Bound<'py, PyAny>,
    lex: &Bound<'py, PyAny>,
    begin: u64,
    identity: &Bound<'py, PyAny>,
) -> PyResult<(i64, i64, i64)> {
    let n: i64 = parents.call_method1("size", (1,))?.extract()?;
    let alphabet = n * (n - 1);
    let mut key = i64::MAX;
    let mut slot = -1_i64;
    let count = parents.len()?;

    let native_result = if count > 0 && parents.getattr("is_cuda")?.extract()? {
        let ops = py
            .import("multigpubeamsearch.cnot_native")?
            .call_method0("_cuda_ops")?;
        if ops.hasattr("local_goal")? {
            Some(ops.call_method1("local_goal", (parents, lex, begin, identity))?)
        } else {
            None
        }
    } else {
        None
    };
    if let Some(result) = native_result.filter(|result| !result.is_none()) {
        (key, slot) = result.extract()?;
    } else if count > 0 {
        let diff = parents.call_method1("__xor__", (identity,))?;
        let changed = diff.call_method1("ne", (0,))?.call_method1("any", (2,))?;
        let target = changed.call_method0("long")?.call_method1("argmax", (1,))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("device", world.getattr("device")?)?;
        let index = py
            .import("torch")?
            .call_method("arange", (count,), Some(&kwargs))?;
        let needed = diff.get_item((&index, &target))?;
        let full = py
            .import("builtins")?
            .getattr("slice")?
            .call1((py.None(), py.None()))?;
        let matches = parents
            .call_method1("__eq__", (needed.get_item((full, py.None()))?,))?
            .call_method1("all", (2,))?;
        matches.set_item((&index, &target), false)?;
        let single = changed
            .call_method1("sum", (1,))?
            .call_method1("__eq__", (1,))?;
        let hits = single
            .call_method1("__and__", (matches.call_method1("any", (1,))?,))?
            .call_method0("nonzero")?
            .call_method0("flatten")?;
        if hits.len()? > 0 {
            let hit = hits.get_item(lex.get_item(&hits)?.call_method0("argmin")?)?;
            let control = matches
                .get_item(&hit)?
                .call_method0("long")?
                .call_method0("argmax")?;
            let target_hit = target.get_item(&hit)?;
            let movement = control
                .call_method1("__mul__", (n - 1,))?
                .call_method1("__add__", (&target_hit,))?
                .call_method1(
                    "__sub__",
                    (target_hit
                        .call_method1("__gt__", (&control,))?
                        .call_method0("long")?,),
                )?;
            key = lex
                .get_item(&hit)?
                .call_method1("__mul__", (alphabet,))?
                .call_method1("__add__", (movement,))?
                .call_method0("item")?
                .extract()?;
            let offset: u64 = hit.call_method0("item")?.extract()?;
            slot = i64::try_from(begin.checked_add(offset).ok_or_else(|| {
                PyOverflowError::new_err("CNOT goal slot exceeds signed 64-bit range")
            })?)
            .map_err(|_| PyOverflowError::new_err("CNOT goal slot exceeds signed 64-bit range"))?;
        }
    }
    Ok((key, slot, alphabet))
}

pub(crate) fn goal<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    parents: &Bound<'py, PyAny>,
    lex: &Bound<'py, PyAny>,
    begin: u64,
    identity: &Bound<'py, PyAny>,
) -> PyResult<Option<(u64, u32)>> {
    let result = local_goal(py, world, parents, lex, begin, identity)
        .and_then(|value| py.check_signals().map(|()| value));
    let (key, slot, alphabet, error) = match result {
        Ok((key, slot, alphabet)) => (key, slot, Some(alphabet), None),
        Err(error) => {
            if (!error.is_instance_of::<PyException>(py)
                && !error.is_instance_of::<PyKeyboardInterrupt>(py))
                || world.getattr("size")?.extract::<usize>()? == 1
            {
                return Err(error);
            }
            let rank: usize = world.getattr("rank")?.extract()?;
            let name: String = error.get_type(py).getattr("__name__")?.extract()?;
            let message = error.value(py).str()?.to_str()?.to_owned();
            (
                i64::MAX,
                -1,
                None,
                Some(format!("rank {rank}: {name}: {message}")),
            )
        }
    };
    let values = vec![key, slot, i64::from(error.is_some())];
    let raw_rows: Vec<Vec<i64>> = world
        .call_method1("gather_fixed", (world.call_method1("longs", (values,))?,))?
        .call_method0("tolist")?
        .extract()?;
    let rows = raw_rows
        .into_iter()
        .map(|row| match row.as_slice() {
            [key, slot, failed] => Ok((*key, *slot, *failed)),
            _ => Err(PyRuntimeError::new_err("invalid CNOT goal exchange")),
        })
        .collect::<PyResult<Vec<_>>>()?;
    if rows.iter().any(|(_, _, failed)| *failed != 0) {
        let size: usize = world.getattr("size")?.extract()?;
        let messages = gather_error_messages_v2(world, size, error.as_deref())?;
        let interrupted = messages
            .iter()
            .any(|message| message.contains(": KeyboardInterrupt:"));
        let details = messages
            .into_iter()
            .filter(|message| !message.is_empty())
            .collect::<Vec<_>>()
            .join(" | ");
        let diagnostic = format!("find CNOT goal failed: {details}");
        if interrupted {
            return Err(PyKeyboardInterrupt::new_err(diagnostic));
        }
        return Err(PyRuntimeError::new_err(diagnostic));
    }
    let (best_key, best_slot) = rows
        .iter()
        .map(|(value, index, _)| (*value, *index))
        .min()
        .ok_or_else(|| PyRuntimeError::new_err("empty CNOT goal exchange"))?;
    if best_slot < 0 {
        return Ok(None);
    }
    let alphabet = alphabet.ok_or_else(|| PyRuntimeError::new_err("missing CNOT goal alphabet"))?;
    if alphabet == 0 {
        return Err(PyZeroDivisionError::new_err("integer modulo by zero"));
    }
    Ok(Some((
        best_slot as u64,
        best_key.rem_euclid(alphabet) as u32,
    )))
}

#[pyfunction]
fn run_cnot_native_goal<'py>(
    py: Python<'py>,
    world: Bound<'py, PyAny>,
    parents: Bound<'py, PyAny>,
    lex: Bound<'py, PyAny>,
    begin: u64,
    identity: Bound<'py, PyAny>,
) -> PyResult<Option<(u64, u32)>> {
    goal(py, &world, &parents, &lex, begin, &identity)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_cnot_native_goal, module)?)
}
