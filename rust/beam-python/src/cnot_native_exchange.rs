//! CNOT record exchange orchestration over the existing transport.
//!
//! PyTorch owns tensors and the caller's process group. The current Stream 5
//! owner, stable packet grouping, collective sequence and empty splits remain
//! unchanged; this module does not turn a GPU address into a Rust reference.
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

pub(crate) fn exchange<'py>(
    py: Python<'py>,
    world: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    targets: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let size: usize = world.getattr("size")?.extract()?;
    if size == 1 {
        return Ok(rows.clone());
    }

    let mut transport = world.getattr("transport")?;
    if !world.getattr("_stream5_library")?.is_none() && transport.is_none() {
        let kwargs = PyDict::new(py);
        kwargs.set_item("prefer_rust", false)?;
        world.call_method("start_transport", (), Some(&kwargs))?;
        transport = world.getattr("transport")?;
    }

    let ops = if rows.getattr("is_cuda")?.is_truthy()? {
        Some(
            py.import("multigpubeamsearch.cnot_native")?
                .call_method0("_cuda_ops")?,
        )
    } else {
        None
    };
    let native_packet = match &ops {
        Some(ops) => ops.hasattr("exchange_packet")?,
        None => false,
    };
    let (send, counts) = if native_packet {
        let packet = ops
            .ok_or_else(|| PyRuntimeError::new_err("CNOT exchange operators unavailable"))?
            .call_method1("exchange_packet", (rows, targets, size))?;
        (packet.get_item(0)?, packet.get_item(1)?)
    } else {
        let torch = py.import("torch")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("minlength", size)?;
        let counts = torch.call_method("bincount", (targets,), Some(&kwargs))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("stable", true)?;
        let order = torch.call_method("argsort", (targets,), Some(&kwargs))?;
        let send = rows
            .call_method1("index_select", (0, order))?
            .call_method0("contiguous")?;
        (send, counts)
    };
    if !transport.is_none() {
        return transport.call_method1("exchange", (send, counts));
    }

    let torch = py.import("torch")?;
    let dist = py.import("torch.distributed")?;
    let recv_counts = torch.call_method1("empty_like", (&counts,))?;
    dist.call_method1("all_to_all_single", (&recv_counts, &counts))?;
    let send_split = counts.call_method0("tolist")?;
    let recv_split = recv_counts.call_method0("tolist")?;
    let received_size = py.import("builtins")?.call_method1("sum", (&recv_split,))?;
    let columns = rows.getattr("shape")?.get_item(1)?;
    let received = rows.call_method1("new_empty", ((received_size, columns),))?;
    dist.call_method1(
        "all_to_all_single",
        (&received, &send, recv_split, send_split),
    )?;
    Ok(received)
}

#[pyfunction]
fn run_cnot_native_exchange<'py>(
    py: Python<'py>,
    world: Bound<'py, PyAny>,
    rows: Bound<'py, PyAny>,
    targets: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    exchange(py, &world, &rows, &targets)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_cnot_native_exchange, module)?)
}
