use crate::runtime_error;
use beam_core::precision::{finite_16le, Float16Format};
use beam_runtime::mlp_snapshot::{MlpSnapshotInput, MlpSnapshotStore};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

#[pyclass(name = "MlpSnapshotStore")]
struct PyMlpSnapshotStore {
    inner: Mutex<MlpSnapshotStore>,
    running: Mutex<()>,
    cancel: Arc<AtomicBool>,
}

#[pymethods]
impl PyMlpSnapshotStore {
    #[new]
    fn new() -> Self {
        Self {
            inner: Mutex::new(MlpSnapshotStore::default()),
            running: Mutex::new(()),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare(
        &self,
        py: Python<'_>,
        source_dir: PathBuf,
        cache_dir: PathBuf,
        run_dir: PathBuf,
        graph_hash: String,
        artifact_hash: String,
        tensors: Vec<String>,
        force_new: bool,
    ) -> PyResult<(PathBuf, bool, u64)> {
        self.prepare_impl(
            py,
            MlpSnapshotInput {
                source_dir,
                cache_dir,
                run_dir,
                graph_hash,
                artifact_hash,
                tensors,
                expected_sizes: None,
            },
            force_new,
            false,
        )
    }

    /// Only for a session that revalidates the snapshot at its launch boundary.
    #[allow(clippy::too_many_arguments)]
    fn prepare_for_session(
        &self,
        py: Python<'_>,
        source_dir: PathBuf,
        cache_dir: PathBuf,
        run_dir: PathBuf,
        graph_hash: String,
        artifact_hash: String,
        tensors: Vec<String>,
        force_new: bool,
    ) -> PyResult<(PathBuf, bool, u64)> {
        self.prepare_impl(
            py,
            MlpSnapshotInput {
                source_dir,
                cache_dir,
                run_dir,
                graph_hash,
                artifact_hash,
                tensors,
                expected_sizes: None,
            },
            force_new,
            true,
        )
    }

    /// The sizes must come from the caller's validated manifest and tensors.
    #[allow(clippy::too_many_arguments)]
    fn prepare_sized_for_session(
        &self,
        py: Python<'_>,
        source_dir: PathBuf,
        cache_dir: PathBuf,
        run_dir: PathBuf,
        graph_hash: String,
        artifact_hash: String,
        tensors: Vec<String>,
        expected_sizes: Vec<u64>,
        force_new: bool,
    ) -> PyResult<(PathBuf, bool, u64)> {
        self.prepare_impl(
            py,
            MlpSnapshotInput {
                source_dir,
                cache_dir,
                run_dir,
                graph_hash,
                artifact_hash,
                tensors,
                expected_sizes: Some(expected_sizes),
            },
            force_new,
            true,
        )
    }

    fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    fn verify_snapshot(
        &self,
        py: Python<'_>,
        path: PathBuf,
        graph_hash: String,
        artifact_hash: String,
        tensors: Vec<String>,
    ) -> PyResult<()> {
        let _running = self.running.try_lock().map_err(|_| {
            PyRuntimeError::new_err("concurrent MLP snapshot preparation is unsupported")
        })?;
        self.cancel.store(false, Ordering::Release);
        crate::interrupt::run(py, &self.cancel, || {
            self.inner
                .lock()
                .map_err(runtime_error)?
                .verify_snapshot_with_cancel(
                    &path,
                    &graph_hash,
                    &artifact_hash,
                    &tensors,
                    &self.cancel,
                )
                .map_err(runtime_error)
        })
    }

    fn clear(&self) -> PyResult<()> {
        let _running = self.running.try_lock().map_err(|_| {
            PyRuntimeError::new_err("cannot clear an active MLP snapshot preparation")
        })?;
        self.inner.lock().map_err(runtime_error)?.clear();
        Ok(())
    }
}

impl PyMlpSnapshotStore {
    fn prepare_impl(
        &self,
        py: Python<'_>,
        input: MlpSnapshotInput,
        force_new: bool,
        defer: bool,
    ) -> PyResult<(PathBuf, bool, u64)> {
        let _running = self.running.try_lock().map_err(|_| {
            PyRuntimeError::new_err("concurrent MLP snapshot preparation is unsupported")
        })?;
        self.cancel.store(false, Ordering::Release);
        crate::interrupt::run(py, &self.cancel, || {
            let mut store = self.inner.lock().map_err(runtime_error)?;
            let snapshot = if defer {
                store.prepare_for_session_with_cancel(&input, force_new, &self.cancel)
            } else {
                store.prepare_with_cancel(&input, force_new, &self.cancel)
            }
            .map_err(runtime_error)?;
            Ok((snapshot.path, snapshot.reused, snapshot.copied_bytes))
        })
    }
}

/// Borrow immutable Python bytes; the reference stays alive while the GIL is released.
#[pyfunction]
fn mlp_finite_16le(py: Python<'_>, data: &Bound<'_, PyBytes>, dtype: &str) -> PyResult<bool> {
    let format = match dtype {
        "fp16" => Float16Format::Fp16,
        "bf16" => Float16Format::Bf16,
        _ => return Err(PyValueError::new_err("expected fp16 or bf16")),
    };
    let bytes = data.as_bytes();
    py.allow_threads(|| finite_16le(bytes, format))
        .map_err(runtime_error)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMlpSnapshotStore>()?;
    module.add_function(wrap_pyfunction!(mlp_finite_16le, module)?)?;
    Ok(())
}
