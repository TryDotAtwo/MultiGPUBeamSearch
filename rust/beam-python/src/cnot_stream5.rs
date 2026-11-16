//! Python access to Rust-owned Stream 5 lifecycle; the active transport is unchanged.
use beam_native_sys::cnot_stream5::{Stream5Communicator, Stream5Library};
use pyo3::{prelude::*, types::PyBytes};
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
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStream5Library>()?;
    module.add_class::<PyStream5Communicator>()
}
