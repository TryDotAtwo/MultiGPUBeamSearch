//! PyTorch adapter for the Rust-owned CNOT checkpoint scorer handle.
use beam_native_sys::cnot_tqp_scorer::TqpScorer;
use pyo3::{
    exceptions::{PyOverflowError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyTuple},
};
use std::path::PathBuf;

#[pyclass(name = "TQPScorer", unsendable)]
pub(crate) struct PyTqpScorer {
    scorer: Option<TqpScorer>,
    device: Py<PyAny>,
    metadata: Py<PyAny>,
    feature_ids: Vec<String>,
    wires: u32,
    batch: i64,
    native_batch: bool,
}

#[pymethods]
impl PyTqpScorer {
    #[new]
    #[pyo3(signature = (library, manifest, *, n, device, inference_batch=4, experimental_native_batch=false))]
    fn new(
        py: Python<'_>,
        library: PathBuf,
        manifest: PathBuf,
        n: i64,
        device: Bound<'_, PyAny>,
        inference_batch: i64,
        experimental_native_batch: bool,
    ) -> PyResult<Self> {
        if !(2..=i64::from(u32::MAX)).contains(&n) {
            return Err(PyValueError::new_err("n must be a valid CNOT dimension"));
        }
        if inference_batch < 1 {
            return Err(PyValueError::new_err("inference_batch must be positive"));
        }
        let torch = py.import("torch")?;
        let version: String = torch.getattr("__version__")?.extract()?;
        if version.split('+').next() != Some("2.9.0")
            || torch.getattr("version")?.getattr("cuda")?.is_none()
        {
            return Err(PyRuntimeError::new_err(
                "the pinned tch 0.22.0 binding requires CUDA PyTorch 2.9.0",
            ));
        }
        let device_object = torch.getattr("device")?.call1((&device,))?;
        let device_type: String = device_object.getattr("type")?.extract()?;
        let device_index: Option<u32> = device_object.getattr("index")?.extract()?;
        if device_type != "cuda" || device_index.is_none() {
            return Err(PyValueError::new_err(
                "select an explicit rank-local CUDA device such as cuda:0",
            ));
        }
        let device_index = device_index.expect("checked explicit CUDA device");
        let cuda_guard = torch
            .getattr("cuda")?
            .call_method1("device", (&device_object,))?;
        let scorer = with_context(py, &cuda_guard, || {
            TqpScorer::open(
                &library,
                &manifest,
                n as u32,
                device_index,
                experimental_native_batch,
            )
            .map_err(crate::runtime_error)
        })?;

        let feature_ids = scorer.feature_ids().to_vec();
        let batch = scorer
            .effective_batch(inference_batch)
            .map_err(crate::runtime_error)?;

        // Preserve the public Python metadata dictionary; Rust has already
        // parsed and validated the scorer contract.
        let metadata = py
            .import("json")?
            .call_method1("loads", (scorer.metadata(),))?;

        Ok(Self {
            scorer: Some(scorer),
            device: device_object.unbind(),
            metadata: metadata.unbind(),
            feature_ids,
            wires: n as u32,
            batch,
            native_batch: experimental_native_batch,
        })
    }

    #[getter]
    fn device(&self, py: Python<'_>) -> Py<PyAny> {
        self.device.clone_ref(py)
    }

    #[getter]
    fn n(&self) -> u32 {
        self.wires
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> Py<PyAny> {
        self.metadata.clone_ref(py)
    }

    #[getter(_batch)]
    fn batch(&self) -> i64 {
        self.batch
    }

    #[getter(_native_batch)]
    fn native_batch(&self) -> bool {
        self.native_batch
    }

    fn close(&mut self) {
        self.scorer.take();
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exception_type: &Bound<'_, PyAny>,
        _exception_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        self.close();
    }

    fn parent_scores<'py>(
        &self,
        py: Python<'py>,
        rows: Bound<'py, PyAny>,
        mode: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.parent_scores_native(py, rows, mode)
    }

    fn child_values<'py>(
        &self,
        py: Python<'py>,
        rows: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.child_values_native(py, rows)
    }
}

impl PyTqpScorer {
    pub(crate) fn parent_scores_native<'py>(
        &self,
        py: Python<'py>,
        rows: Bound<'py, PyAny>,
        mode: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let head = match mode {
            "qhat" => 0,
            "proposal" => 1,
            _ => {
                return Err(PyValueError::new_err(
                    "parent scorer mode must be qhat or proposal",
                ))
            }
        };
        self.score(py, rows, head)
    }

    pub(crate) fn child_values_native<'py>(
        &self,
        py: Python<'py>,
        rows: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.score(py, rows, 2)
    }

    fn score<'py>(
        &self,
        py: Python<'py>,
        rows: Bound<'py, PyAny>,
        head: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let torch = py.import("torch")?;
        if !torch
            .call_method1("is_tensor", (&rows,))?
            .extract::<bool>()?
        {
            return Err(PyValueError::new_err(
                "rows must be rank-local CUDA int64 [batch, n, ceil(n/64)]",
            ));
        }
        let dtype_ok = py
            .import("operator")?
            .call_method1("eq", (rows.getattr("dtype")?, torch.getattr("int64")?))?
            .extract::<bool>()?;
        let device_ok = py
            .import("operator")?
            .call_method1("eq", (rows.getattr("device")?, self.device.bind(py)))?
            .extract::<bool>()?;
        let shape: Vec<usize> = rows.getattr("shape")?.extract()?;
        let words = (usize::try_from(self.wires)
            .map_err(|_| PyOverflowError::new_err("CNOT dimension overflow"))?
            + 63)
            / 64;
        if !dtype_ok
            || !device_ok
            || shape.len() != 3
            || shape[1] != self.wires as usize
            || shape[2] != words
        {
            return Err(PyValueError::new_err(
                "rows must be rank-local CUDA int64 [batch, n, ceil(n/64)]",
            ));
        }
        let count = i64::try_from(shape[0])
            .map_err(|_| PyOverflowError::new_err("CNOT packet size overflow"))?;
        let columns = if head == 2 {
            1_usize
        } else {
            usize::try_from(self.wires)
                .ok()
                .and_then(|n| n.checked_mul(n.saturating_sub(1)))
                .ok_or_else(|| PyOverflowError::new_err("CNOT score width overflow"))?
        };
        let output_shape = if head == 2 {
            vec![shape[0]]
        } else {
            vec![shape[0], columns]
        };
        let cuda_guard = torch
            .getattr("cuda")?
            .call_method1("device", (self.device.bind(py),))?;
        with_context(py, &cuda_guard, || {
            let inference_guard = torch.getattr("inference_mode")?.call0()?;
            with_context(py, &inference_guard, || {
                let scorer = self
                    .scorer
                    .as_ref()
                    .ok_or_else(|| PyRuntimeError::new_err("CNOT scorer is closed"))?;
                if scorer.is_closed() {
                    return Err(PyRuntimeError::new_err("CNOT scorer is closed"));
                }
                self.score_validated(py, &torch, &rows, head, count, output_shape)
            })
        })
    }

    fn score_validated<'py>(
        &self,
        py: Python<'py>,
        torch: &Bound<'py, PyAny>,
        rows: &Bound<'py, PyAny>,
        head: u32,
        count: i64,
        output_shape: Vec<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let output_shape = PyTuple::new(py, output_shape)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", torch.getattr("float32")?)?;
        kwargs.set_item("device", self.device.bind(py))?;
        let output = torch
            .getattr("empty")?
            .call((output_shape,), Some(&kwargs))?;
        if count == 0 {
            return Ok(output);
        }

        let contiguous_rows = rows.call_method0("contiguous")?;
        let status_shape = PyTuple::new(
            py,
            [usize::try_from(count)
                .map_err(|_| PyOverflowError::new_err("CNOT packet size overflow"))?],
        )?;
        let status_kwargs = PyDict::new(py);
        status_kwargs.set_item("dtype", torch.getattr("int64")?)?;
        status_kwargs.set_item("device", self.device.bind(py))?;
        let status = torch
            .getattr("empty")?
            .call((status_shape,), Some(&status_kwargs))?;
        let scorer = self
            .scorer
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CNOT scorer is closed"))?;
        scorer.score_tensors(py, head, self.batch, &contiguous_rows, &output, &status)?;
        self.check_scores(py, &output, &status, count)?;
        Ok(output)
    }

    fn check_scores(
        &self,
        py: Python<'_>,
        output: &Bound<'_, PyAny>,
        status: &Bound<'_, PyAny>,
        count: i64,
    ) -> PyResult<()> {
        let finite = output
            .call_method0("isfinite")?
            .call_method1("reshape", (count, -1))?
            .call_method1("all", (1,))?;
        let failed = py.import("operator")?.call_method1(
            "or_",
            (
                status.call_method1("ne", (0,))?,
                py.import("operator")?.call_method1("invert", (finite,))?,
            ),
        )?;
        let any_failed: bool = failed
            .call_method0("any")?
            .call_method0("item")?
            .extract()?;
        if !any_failed {
            return Ok(());
        }
        let row: usize = failed
            .call_method0("int")?
            .call_method0("argmax")?
            .call_method0("item")?
            .extract()?;
        let coverage: i64 = status.get_item(row)?.call_method0("item")?.extract()?;
        if coverage != 0 {
            let slot = coverage.div_euclid(256);
            let value = coverage.rem_euclid(256);
            if slot > 0 {
                if let Some(feature) = self.feature_ids.get((slot - 1) as usize) {
                    return Err(PyRuntimeError::new_err(format!(
                        "CNOT row {row}: feature {feature} rejected (status {value})"
                    )));
                }
            }
            return Err(PyRuntimeError::new_err(format!(
                "CNOT row {row}: invalid feature coverage status {coverage}"
            )));
        }
        Err(PyRuntimeError::new_err(format!(
            "CNOT row {row}: nonfinite model score"
        )))
    }
}

fn with_context<T>(
    py: Python<'_>,
    context: &Bound<'_, PyAny>,
    operation: impl FnOnce() -> PyResult<T>,
) -> PyResult<T> {
    context.call_method0("__enter__")?;
    let result = operation();
    let exit = context.call_method1("__exit__", (py.None(), py.None(), py.None()));
    match result {
        Ok(value) => {
            exit?;
            Ok(value)
        }
        Err(error) => {
            let _ = exit;
            Err(error)
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTqpScorer>()
}
