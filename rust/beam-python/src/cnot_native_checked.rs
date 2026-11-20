//! Rank-wide error coordination for fallible CNOT stages.
//!
//! The operation still runs on the calling thread. Checked-stage votes and
//! diagnostics use the existing CPU/Gloo control group so a CUDA allocation or
//! NCCL failure does not require another GPU collective.
use pyo3::{
    exceptions::{PyBaseException, PyOverflowError, PyRuntimeError},
    prelude::*,
    types::{PyDict, PyList},
};

fn cpu_i64_tensor<'py>(py: Python<'py>, values: Vec<i64>) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import("torch")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", torch.getattr("int64")?)?;
    kwargs.set_item("device", "cpu")?;
    torch.call_method("tensor", (values,), Some(&kwargs))
}

pub(crate) fn gather_error_messages_v2<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    size: usize,
    error: Option<&str>,
) -> PyResult<Vec<String>> {
    let local_len = error.map_or(0, str::len);
    let local_len = i64::try_from(local_len)
        .map_err(|_| PyOverflowError::new_err("CNOT error message exceeds int64 length"))?;
    let length_tensor = cpu_i64_tensor(py, vec![local_len])?;
    let gathered_lengths = world
        .call_method1("_control_gather", (length_tensor,))?
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
    let payload_tensor = cpu_i64_tensor(py, local_bytes)?;
    let gathered_payload = world
        .call_method1("_control_gather", (payload_tensor,))?
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
            if !error.is_instance_of::<PyBaseException>(py) {
                return Err(error);
            }
            let rank: usize = world.getattr("rank")?.extract()?;
            let name: String = error.get_type(py).getattr("__name__")?.extract()?;
            let message = error.value(py).str()?.to_str()?.to_owned();
            (None, Some(format!("rank {rank}: {name}: {message}")))
        }
    };
    let any_failed: bool = world
        .call_method1("any_rank_failed", (error.is_some(),))?
        .extract()?;
    if any_failed {
        let messages = if byte_gather {
            gather_error_messages_v2(py, world, size, error.as_deref())?
        } else {
            let errors = PyList::new(py, (0..size).map(|_| py.None()))?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("group", world.getattr("_control_group")?)?;
            py.import("torch.distributed")?.call_method(
                "all_gather_object",
                (&errors, error),
                Some(&kwargs),
            )?;
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

/// Coordinate Rust-owned CNOT stages through the existing Gloo control group.
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
    checked_v2(py, world, stage, result)
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

#[pyfunction]
fn run_cnot_native_checked_control<'py>(
    py: Python<'py>,
    world: Bound<'py, PyAny>,
    stage: &str,
    callback: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    checked_v2(py, &world, stage, callback.call0())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_cnot_native_checked, module)?)?;
    module.add_function(wrap_pyfunction!(run_cnot_native_checked_v2, module)?)?;
    module.add_function(wrap_pyfunction!(run_cnot_native_checked_control, module)?)
}
