//! Coarse Python adapter for Rust-owned CNOT Stream 5 exchanges.
use beam_native_sys::cnot_stream5::{Stream5Communicator, Stream5Library};
use beam_runtime::cnot_exchange::{self, Backend, Counts};
use pyo3::{
    exceptions::{PyOverflowError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use std::path::PathBuf;

#[pyclass(name = "CnotStream5Library", unsendable)]
struct PyStream5Library {
    inner: Stream5Library,
}

#[pymethods]
impl PyStream5Library {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: Stream5Library::load(&path).map_err(crate::runtime_error)?,
        })
    }

    fn unique_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.unique_id().map_err(crate::runtime_error)?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[getter]
    fn supports_exchange(&self) -> bool {
        self.inner.supports_exchange()
    }

    #[getter]
    fn supports_collectives(&self) -> bool {
        self.inner.supports_collectives()
    }

    #[getter]
    fn id_size(&self) -> usize {
        self.inner.id_size()
    }

    fn init(
        &self,
        id: &Bound<'_, PyBytes>,
        rank: u32,
        world: u32,
    ) -> PyResult<PyStream5Communicator> {
        let inner = self
            .inner
            .init(id.as_bytes(), rank, world)
            .map_err(crate::runtime_error)?;
        Ok(PyStream5Communicator { inner })
    }
}

#[pyclass(name = "CnotStream5Communicator", unsendable)]
struct PyStream5Communicator {
    inner: Stream5Communicator,
}

#[pymethods]
impl PyStream5Communicator {
    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_closed()
    }

    #[pyo3(signature = (abort=false))]
    fn close(&mut self, abort: bool) -> PyResult<()> {
        self.inner.close(abort).map_err(crate::runtime_error)
    }

    fn validate(
        &self,
        send: &Bound<'_, PyAny>,
        counts: &Bound<'_, PyAny>,
        world: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.inner.validate_tensors(send, counts, world)
    }

    fn all_reduce_i64(&self, tensor: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.all_reduce_i64(tensor)
    }

    fn all_gather_i64<'py>(&self, tensor: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.all_gather_i64(tensor)
    }

    fn validate_all_gather_i64_into(
        &self,
        tensor: &Bound<'_, PyAny>,
        work: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.inner
            .validate_all_gather_i64_into(tensor, work, output)
    }

    fn all_gather_i64_into<'py>(
        &self,
        tensor: &Bound<'py, PyAny>,
        work: &Bound<'py, PyAny>,
        output: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.all_gather_i64_into(tensor, work, output)
    }

    fn broadcast_i64(&self, tensor: &Bound<'_, PyAny>, source: u32) -> PyResult<()> {
        self.inner.broadcast_i64(tensor, source)
    }

    fn exchange<'py>(
        &self,
        send: Bound<'py, PyAny>,
        counts: Bound<'py, PyAny>,
        world: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.validate_tensors(&send, &counts, &world)?;
        let mut stages = TensorExchange {
            comm: &self.inner,
            send,
            counts,
            world,
            backing: None,
            source: None,
        };
        cnot_exchange::exchange(&mut stages)
    }
}

struct TensorExchange<'a, 'py> {
    comm: &'a Stream5Communicator,
    send: Bound<'py, PyAny>,
    counts: Bound<'py, PyAny>,
    world: Bound<'py, PyAny>,
    backing: Option<Bound<'py, PyAny>>,
    source: Option<Bound<'py, PyAny>>,
}

impl<'py> Backend for TensorExchange<'_, 'py> {
    type Error = PyErr;
    type Output = Bound<'py, PyAny>;

    fn send_count(&mut self) -> PyResult<u64> {
        self.send.call_method1("size", (0,))?.extract()
    }

    fn any_overflow(&mut self, local: bool) -> PyResult<bool> {
        self.world
            .call_method1("any_rank_failed", (local,))?
            .extract()
    }

    fn prepare_rows(&mut self, size: u32) -> PyResult<()> {
        let width: u64 = self.send.call_method1("size", (1,))?.extract()?;
        let backing = self
            .send
            .call_method1("new_empty", ((u64::from(size).max(1), width),))?;
        let source = if self.send_count()? == 0 {
            self.send.call_method1("new_empty", ((1_u64, width),))?
        } else {
            self.send.clone()
        };
        self.backing = Some(backing);
        self.source = Some(source);
        Ok(())
    }

    fn any_prepare_failure(&mut self, local: bool) -> PyResult<bool> {
        self.world
            .call_method1("any_rank_failed", (local,))?
            .extract()
    }

    fn peer_prepare_failure(&mut self) -> PyErr {
        PyRuntimeError::new_err(
            "CNOT receive buffer allocation failed on another rank before payload exchange",
        )
    }

    fn exchange_counts(&mut self) -> PyResult<Counts> {
        let torch = self.send.py().import("torch")?;
        let send_counts = self.counts.call_method1("to", (torch.getattr("int32")?,))?;
        let recv_counts = torch.call_method1("empty_like", (&send_counts,))?;
        self.comm
            .exchange_counts_tensors(&send_counts, &recv_counts)?;
        let send_raw: Vec<i64> = self.counts.call_method0("tolist")?.extract()?;
        let recv_raw: Vec<i64> = recv_counts.call_method0("tolist")?.extract()?;
        let mut invalid_send = false;
        let send = send_raw
            .into_iter()
            .map(|n| {
                if !(0..=i64::from(u32::MAX)).contains(&n) {
                    invalid_send = true;
                    0
                } else {
                    n as u32
                }
            })
            .collect();
        let receive = recv_raw.into_iter().map(|n| n as i32 as u32).collect();
        Ok(Counts {
            send,
            receive,
            invalid_send,
        })
    }

    fn exchange_rows(
        &mut self,
        send: &[u32],
        receive: &[u32],
        size: u32,
    ) -> PyResult<Self::Output> {
        let backing = self.backing.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("CNOT receive buffer was not prepared before payload exchange")
        })?;
        let source = self.source.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("CNOT send buffer was not prepared before payload exchange")
        })?;
        self.comm
            .exchange_row_tensors(source, backing, send, receive)?;
        backing.call_method1("narrow", (0, 0, size))
    }

    fn send_overflow(&mut self) -> PyErr {
        PyOverflowError::new_err("per-rank CNOT send exceeds Stream 5 uint32 offsets")
    }
    fn receive_overflow(&mut self) -> PyErr {
        PyOverflowError::new_err(
            "per-rank CNOT receive exceeds Stream 5 uint32 offsets or peer counts are invalid",
        )
    }
    fn invalid_counts(&mut self) -> PyErr {
        PyValueError::new_err("CNOT send counts are invalid or do not match the record count")
    }
}

#[pyfunction]
pub(crate) fn run_cnot_native_start_transport<'py>(
    py: Python<'py>,
    world: Bound<'py, PyAny>,
    prefer_rust: bool,
) -> PyResult<()> {
    let native_library = world.getattr("_stream5_library")?;
    if native_library.is_none() {
        return Ok(());
    }

    let module = py.import("multigpubeamsearch.cnot_stream5")?;
    let rust_transport = module.getattr("RustStream5")?;
    let python_transport = module.getattr("Stream5")?;
    let library = if prefer_rust {
        rust_transport.call_method1("compatible_library", (&native_library,))?
    } else {
        py.None().into_bound(py)
    };

    let use_rust = if prefer_rust {
        let supported = if library.is_none() { 0_i64 } else { 1_i64 };
        let local = world.call_method1("longs", (vec![supported],))?;
        let total: usize = world
            .call_method1("sum", (local,))?
            .call_method0("item")?
            .extract()?;
        total == world.getattr("size")?.extract::<usize>()?
    } else {
        false
    };

    let transport = if use_rust {
        rust_transport.call1((library, world.clone()))?
    } else {
        python_transport.call1((&native_library, world.clone()))?
    };
    world.setattr("transport", transport)?;
    Ok(())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStream5Library>()?;
    module.add_class::<PyStream5Communicator>()?;
    module.add_function(wrap_pyfunction!(run_cnot_native_start_transport, module)?)
}
