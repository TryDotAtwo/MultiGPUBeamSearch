//! Coarse Python adapter for the owned packed CNOT input contract.
use beam_runtime::cnot_search::{self, Backend, Edge, Goal, Options, Round};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict, PyList, PyTuple},
};

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

struct PyStages<'py> {
    py: Python<'py>,
    stages: Bound<'py, PyAny>,
}

impl Backend for PyStages<'_> {
    type Error = PyErr;
    type Layer = Py<PyAny>;
    type Records = Py<PyAny>;

    fn initialize(&mut self) -> PyResult<()> {
        self.stages.call_method0("initialize")?;
        Ok(())
    }
    fn goal(&mut self, begin: u64) -> PyResult<Option<Goal>> {
        let value = self.stages.call_method1("goal", (begin,))?;
        let result: Option<(u64, u32)> = value.extract()?;
        Ok(result.map(|(slot, movement)| Goal { slot, movement }))
    }
    fn expand(&mut self, begin: u64, k: u32) -> PyResult<()> {
        self.stages.call_method1("expand", (begin, k))?;
        Ok(())
    }
    fn exchange_candidates(&mut self) -> PyResult<()> {
        self.stages.call_method0("exchange_candidates")?;
        Ok(())
    }
    fn deduplicate_and_score(&mut self) -> PyResult<()> {
        self.stages.call_method0("deduplicate_and_score")?;
        Ok(())
    }
    fn select(&mut self, width: u64) -> PyResult<()> {
        self.stages.call_method1("select", (width,))?;
        Ok(())
    }
    fn redistribute(&mut self, depth: u64) -> PyResult<Round<Self::Layer, Self::Records>> {
        let (count, layer, records) = self
            .stages
            .call_method1("redistribute", (depth,))?
            .extract()?;
        Ok(Round {
            count,
            layer,
            records,
        })
    }
    fn history_edge(
        &mut self,
        records: &Self::Records,
        target: u32,
        local_index: u64,
    ) -> PyResult<Edge> {
        let (parent, route, owner) = self
            .stages
            .call_method1("history_edge", (records.bind(self.py), target, local_index))?
            .extract()?;
        Ok(Edge {
            parent,
            route,
            owner,
        })
    }
    fn verifies_construction(&mut self, gates: &[u32]) -> PyResult<bool> {
        self.stages
            .call_method1("verifies_construction", (gates.to_vec(),))?
            .extract()
    }
    fn invalid(&self, message: &'static str) -> PyErr {
        PyRuntimeError::new_err(message)
    }
}

/// Run on the calling rank's thread: Torch's current device/stream and
/// inference-mode context are thread-local. Only complete stages cross PyO3.
#[pyfunction]
fn run_cnot<'py>(
    py: Python<'py>,
    stages: Bound<'py, PyAny>,
    rank: u32,
    world: u32,
    width: u64,
    shortlist: u32,
    alphabet: u32,
    max_depth: u64,
) -> PyResult<Py<PyAny>> {
    let mut backend = PyStages { py, stages };
    let outcome = cnot_search::search(
        &mut backend,
        Options {
            rank,
            world,
            width,
            shortlist,
            alphabet,
            max_depth,
        },
    )?;
    let gates = match outcome.gates {
        Some(gates) => PyTuple::new(py, gates)?.into_any().unbind(),
        None => py.None(),
    };
    let layers = PyList::new(py, outcome.layers)?;
    let routes = PyList::empty(py);
    for route in outcome.routes {
        let item = PyDict::new(py);
        item.set_item("source", route.source)?;
        item.set_item("hash_owner", route.hash_owner)?;
        item.set_item("target", route.target)?;
        item.set_item("move", route.movement)?;
        routes.append(item)?;
    }
    Ok(py
        .import("multigpubeamsearch.cnot")?
        .getattr("CnotResult")?
        .call1((outcome.status, gates, layers, routes))?
        .unbind())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCnotMatrix>()?;
    module.add_function(wrap_pyfunction!(run_cnot, module)?)
}
