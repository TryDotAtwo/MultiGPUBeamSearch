//! Coarse Python adapter for Rust-owned CNOT Stream 5 exchanges.
use beam_native_sys::cnot_stream5::{Stream5Communicator, Stream5Library};
use beam_runtime::cnot_exchange::{self, Backend, Counts};
use pyo3::{
    exceptions::{PyOverflowError, PyValueError},
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

    fn all_gather_i64<'py>(
        &self,
        tensor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.all_gather_i64(tensor)
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
        };
        cnot_exchange::exchange(&mut stages)
    }
}

struct TensorExchange<'a, 'py> {
    comm: &'a Stream5Communicator,
    send: Bound<'py, PyAny>,
    counts: Bound<'py, PyAny>,
    world: Bound<'py, PyAny>,
}

impl<'py> Backend for TensorExchange<'_, 'py> {
    type Error = PyErr;
    type Output = Bound<'py, PyAny>;

    fn send_count(&mut self) -> PyResult<u64> {
        self.send.call_method1("size", (0,))?.extract()
    }

    fn any_overflow(&mut self, local: bool) -> PyResult<bool> {
        let flag = self
            .world
            .call_method1("longs", (vec![if local { 1_i64 } else { 0_i64 }],))?;
        let total = self.world.call_method1("sum", (flag,))?;
        Ok(total.call_method0("item")?.extract::<u64>()? != 0)
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
        let width: u64 = self.send.call_method1("size", (1,))?.extract()?;
        let backing = self
            .send
            .call_method1("new_empty", ((u64::from(size).max(1), width),))?;
        let source = if self.send_count()? == 0 {
            self.send.call_method1("new_empty", ((1_u64, width),))?
        } else {
            self.send.clone()
        };
        self.comm
            .exchange_row_tensors(&source, &backing, send, receive)?;
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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStream5Library>()?;
    module.add_class::<PyStream5Communicator>()
}
