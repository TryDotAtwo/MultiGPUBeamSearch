use crate::runtime_error;
use beam_runtime::mlp_snapshot::{MlpSnapshotInput, MlpSnapshotStore};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
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
            },
            force_new,
            true,
        )
    }

    fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
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

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMlpSnapshotStore>()?;
    Ok(())
}
