//! Rank-wide error coordination for fallible CNOT stages.
//!
//! The operation still runs on the calling thread. The v1 ABI retains its
//! object gather for compatibility; the v2 path gathers UTF-8 diagnostics as
//! fixed-size tensors over the existing world transport. Torch owns buffers.
use pyo3::{
    exceptions::{PyException, PyOverflowError, PyRuntimeError},
    prelude::*,
    types::PyList,
};

fn gather_error_messages_v2<'py>(
    world: &Bound<'py, PyAny>,
    size: usize,
    error: Option<&str>,
) -> PyResult<Vec<String>> {
    let local_len = error.map_or(0, str::len);
    let local_len = i64::try_from(local_len)
        .map_err(|_| PyOverflowError::new_err("CNOT error message exceeds int64 length"))?;
    let length_tensor = world.call_method1("longs", (vec![local_len],))?;
    let gathered_lengths = world
        .call_method1("gather_fixed", (length_tensor,))?
        .call_method0("flatten")?
        .call_method0("tolist")?;
    let raw_lengths: Vec<i64> = gathered_lengths.extract()?;
    if raw_lengths.len() != size {
        return Err(PyRuntimeError::new_err(
            "CNOT error length gather returned an invalid rank count",
        ));
    }
    let lengths: Vec<usize> = raw_lengths
        .into_iter()
        .map(|length| {
            usize::try_from(length)
                .map_err(|_| PyRuntimeError::new_err("CNOT error length is negative"))
        })
        .collect::<PyResult<_>>()?;
    let max_len = lengths.iter().copied().max().unwrap_or(0);

    let mut local_bytes: Vec<i64> = error
        .unwrap_or_default()
        .as_bytes()
        .iter()
        .copied()
        .map(i64::from)
        .collect();
    if local_bytes.len() > max_len {
        return Err(PyRuntimeError::new_err(
            "CNOT local error exceeds the gathered maximum length",
        ));
    }
    local_bytes.resize(max_len, 0);
    let payload_tensor = world.call_method1("longs", (local_bytes,))?;
    let gathered_payload = world
        .call_method1("gather_fixed", (payload_tensor,))?
        .call_method0("tolist")?;
    let rows: Vec<Vec<i64>> = gathered_payload.extract()?;
    if rows.len() != size {
        return Err(PyRuntimeError::new_err(
            "CNOT error payload gather returned an invalid rank count",
        ));
    }

    lengths
        .into_iter()
        .zip(rows)
        .map(|(length, row)| {
            if length > row.len() {
                return Err(PyRuntimeError::new_err(
                    "CNOT error payload is shorter than its gathered length",
                ));
            }
            let bytes: Vec<u8> = row[..length]
                .iter()
                .map(|value| {
                    u8::try_from(*value)
                        .map_err(|_| PyRuntimeError::new_err("invalid CNOT error payload byte"))
                })
                .collect::<PyResult<_>>()?;
            String::from_utf8(bytes)
                .map_err(|_| PyRuntimeError::new_err("CNOT error payload is not valid UTF-8"))
        })
        .collect()
}

fn checked_impl<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    stage: &str,
    result: PyResult<Bound<'py, PyAny>>,
    byte_gather: bool,
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
        let messages = if byte_gather {
            gather_error_messages_v2(world, size, error.as_deref())?
        } else {
            let errors = PyList::new(py, (0..size).map(|_| py.None()))?;
            py.import("torch.distributed")?
                .call_method1("all_gather_object", (&errors, error))?;
            let gathered: Vec<Option<String>> = errors.extract()?;
            gathered.into_iter().flatten().collect()
        };
        return Err(PyRuntimeError::new_err(format!(
            "{stage} failed: {}",
            messages
                .into_iter()
                .filter(|message| !message.is_empty())
                .collect::<Vec<_>>()
                .join(" | ")
        )));
    }
    Ok(value.unwrap_or_else(|| py.None().into_bound(py)))
}

pub(crate) fn checked<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    stage: &str,
    result: PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    checked_impl(py, world, stage, result, false)
}

pub(crate) fn checked_v2<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    stage: &str,
    result: PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    checked_impl(py, world, stage, result, true)
}

/// Preserve the rank-wide checker negotiation used by Python `World.checked`,
/// then coordinate an operation result without calling a Python stage callback.
pub(crate) fn checked_compat<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    stage: &str,
    result: PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let size: usize = world.getattr("size")?.extract()?;
    if size == 1 {
        return result;
    }

    let mut use_v2 = false;
    let selected = world.getattr("_checked_error_gather_v2")?;
    if selected.is_none() {
        let extension = py
            .import("multigpubeamsearch.rust_control")?
            .call_method0("extension")?;
        let builtins = py.import("builtins")?;
        let checker = builtins.getattr("getattr")?.call1((
            extension,
            "run_cnot_native_checked_v2",
            py.None(),
        ))?;
        let callable = builtins.getattr("callable")?.call1((&checker,))?;
        let supported: bool = world
            .call_method1("all_ranks_support", (&callable,))?
            .extract()?;
        if supported {
            world.setattr("_checked_error_gather_v2", &checker)?;
            use_v2 = true;
        } else {
            world.setattr("_checked_error_gather_v2", false)?;
        }
    } else {
        use_v2 = selected.is_callable();
    }

    if use_v2 {
        checked_v2(py, world, stage, result)
    } else {
        checked(py, world, stage, result)
    }
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

#[pyfunction]
fn run_cnot_native_checked_v2<'py>(
    py: Python<'py>,
    world: Bound<'py, PyAny>,
    stage: &str,
    callback: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    checked_v2(py, &world, stage, callback.call0())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_cnot_native_checked, module)?)?;
    module.add_function(wrap_pyfunction!(run_cnot_native_checked_v2, module)?)
}
