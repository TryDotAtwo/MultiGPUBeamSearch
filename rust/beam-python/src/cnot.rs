//! Coarse Python adapter for the owned packed CNOT input contract.
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

#[pyclass(name = "CnotMatrix", frozen)]
struct PyCnotMatrix {
    inner: beam_core::cnot::CnotMatrix,
}

#[pymethods]
impl PyCnotMatrix {
    #[new]
    fn new(wires: u32, packed: &Bound<'_, PyBytes>) -> PyResult<Self> {
        // The Python byte slice is borrowed only for this call. beam-core owns
        // the decoded words before this method returns.
        let inner = beam_core::cnot::CnotMatrix::from_le_bytes(wires, packed.as_bytes())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    fn verifies_construction(&self, py: Python<'_>, moves: Vec<u32>) -> PyResult<bool> {
        py.allow_threads(|| self.inner.verifies_construction(&moves))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCnotMatrix>()
}
