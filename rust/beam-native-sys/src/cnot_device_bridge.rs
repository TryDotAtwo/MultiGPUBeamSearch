//! Rank-local borrowed CUDA bridge for the CNOT full-matrix distinct stage.
//!
//! This opt-in transition keeps Torch tensor storage and the existing rank
//! collectives. The domain remains on the search's calling thread; no CUDA
//! address becomes a Rust reference or leaves the rank process.
use crate::cnot_domain::{Config, DeviceUnseenWorkspace, DeviceView, Domain, DomainLibrary};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};
use std::{ffi::c_void, path::Path};

fn native_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

pub struct DeviceDomain {
    library: DomainLibrary,
    domain: Option<Domain>,
    device: i32,
    n: i32,
    world: i32,
    closed_capacity: u64,
    child_capacity: u64,
}

impl DeviceDomain {
    pub fn load(path: &Path, device: i32, n: i32, world: i32) -> PyResult<Self> {
        // SAFETY: the explicit library path is supplied by the caller as a
        // trusted native CNOT domain implementation of the documented ABI.
        let library = unsafe { DomainLibrary::load(path) }.map_err(native_error)?;
        if !library.supports_device_expansion() || !library.supports_device_unseen() {
            return Err(PyRuntimeError::new_err(
                "configured CNOT domain library lacks borrowed-device selection",
            ));
        }
        Ok(Self {
            library,
            domain: None,
            device,
            n,
            world,
            closed_capacity: 0,
            child_capacity: 0,
        })
    }

    fn domain(&mut self, closed: u64, children: u64) -> PyResult<&mut Domain> {
        if self.domain.is_none() || closed > self.closed_capacity || children > self.child_capacity
        {
            let config = Config {
                device: self.device,
                n: self.n,
                world: self.world,
                parent_capacity: 0,
                child_capacity: children,
                closed_capacity: closed,
            };
            let next = self.library.create_device(config).map_err(native_error)?;
            if let Some(previous) = self.domain.take() {
                previous.close().map_err(native_error)?;
            }
            self.domain = Some(next);
            self.closed_capacity = closed;
            self.child_capacity = children;
        }
        self.domain
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("CNOT device domain is unavailable"))
    }

    pub fn first_unseen<'py>(
        &mut self,
        py: Python<'py>,
        closed: &Bound<'py, PyAny>,
        children: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let torch = py.import("torch")?;
        exact_tensor(&torch, closed)?;
        exact_tensor(&torch, children)?;
        let device = children.getattr("device")?;
        let closed = closed.call_method0("contiguous")?;
        let children = children.call_method0("contiguous")?;
        let closed_count = check_matrices(py, &closed, self.device, self.n)?;
        let child_count = check_matrices(py, &children, self.device, self.n)?;
        if child_count == 0 {
            return empty(py, &torch, &device, "int64", 0);
        }
        let total = closed_count
            .checked_add(child_count)
            .ok_or_else(|| PyRuntimeError::new_err("CNOT ordinal count overflow"))?;
        let n = u64::try_from(self.n)
            .map_err(|_| PyRuntimeError::new_err("invalid CNOT qubit count"))?;
        let stride = n
            .checked_mul((n + 63) / 64)
            .ok_or_else(|| PyRuntimeError::new_err("CNOT matrix stride overflow"))?;
        let closed_words = closed_count
            .checked_mul(stride)
            .ok_or_else(|| PyRuntimeError::new_err("CNOT closed word count overflow"))?;
        let child_words = child_count
            .checked_mul(stride)
            .ok_or_else(|| PyRuntimeError::new_err("CNOT child word count overflow"))?;
        let device_id = self.device;
        let domain = self.domain(closed_count, child_count)?;
        let bytes = domain
            .unseen_scratch_bytes(closed_count, child_count)
            .map_err(native_error)?;
        let ordinals = empty(py, &torch, &device, "int32", total)?;
        let flags = empty(py, &torch, &device, "uint8", child_count)?;
        let selected = empty(py, &torch, &device, "int64", child_count)?;
        let count = empty(py, &torch, &device, "int32", 1)?;
        let scratch = empty(py, &torch, &device, "uint8", bytes)?;
        let workspace = DeviceUnseenWorkspace {
            ordinals: tensor_view(&torch, &ordinals, device_id, "int32", total)?,
            flags: tensor_view(&torch, &flags, device_id, "uint8", child_count)?,
            selected: tensor_view(&torch, &selected, device_id, "int64", child_count)?,
            count: tensor_view(&torch, &count, device_id, "int32", 1)?,
            scratch: tensor_view(&torch, &scratch, device_id, "uint8", bytes)?,
        };
        let cuda = torch.getattr("cuda")?;
        let current_stream = cuda.call_method1("current_stream", (&device,))?;
        if !current_stream
            .get_type()
            .as_any()
            .is(&cuda.getattr("Stream")?)
            || current_stream
                .getattr("device")?
                .getattr("index")?
                .extract::<i32>()?
                != device_id
        {
            return Err(PyRuntimeError::new_err(
                "CNOT domain requires a plain CUDA stream",
            ));
        }
        let stream: usize = current_stream.getattr("cuda_stream")?.extract()?;
        // SAFETY: both inputs were checked as contiguous int64 matrices on
        // this device. Torch created each workspace allocation separately,
        // retains it in the bound values above and provides its actual numel.
        // All inputs and temporary tensors are on Torch's current CUDA stream;
        // the native operation completes that stream before any value drops.
        unsafe {
            domain.first_unseen_device(
                tensor_view(&torch, &closed, device_id, "int64", closed_words)?,
                closed_count,
                tensor_view(&torch, &children, device_id, "int64", child_words)?,
                child_count,
                &workspace,
                stream as *mut c_void,
            )
        }
        .map_err(native_error)?;
        let found: i64 = count.call_method0("item")?.extract()?;
        if found < 0 || found as u64 > child_count {
            return Err(PyRuntimeError::new_err(
                "invalid CNOT device selection count",
            ));
        }
        selected.call_method1("narrow", (0, 0, found))
    }
}

fn empty<'py>(
    py: Python<'py>,
    torch: &Bound<'py, PyModule>,
    device: &Bound<'py, PyAny>,
    dtype: &str,
    count: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let count =
        i64::try_from(count).map_err(|_| PyRuntimeError::new_err("CNOT tensor size overflow"))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("device", device)?;
    kwargs.set_item("dtype", torch.getattr(dtype)?)?;
    torch.call_method("empty", ((count,),), Some(&kwargs))
}

fn exact_tensor(torch: &Bound<'_, PyModule>, tensor: &Bound<'_, PyAny>) -> PyResult<()> {
    if !tensor.get_type().as_any().is(&torch.getattr("Tensor")?) {
        return Err(PyRuntimeError::new_err(
            "CNOT device domain requires a plain torch.Tensor",
        ));
    }
    Ok(())
}

fn tensor_view(
    torch: &Bound<'_, PyModule>,
    tensor: &Bound<'_, PyAny>,
    device: i32,
    dtype: &str,
    expected: u64,
) -> PyResult<DeviceView> {
    exact_tensor(torch, tensor)?;
    if !tensor.getattr("is_cuda")?.extract::<bool>()?
        || !tensor.getattr("dtype")?.is(&torch.getattr(dtype)?)
        || tensor
            .getattr("device")?
            .getattr("index")?
            .extract::<i32>()?
            != device
        || !tensor.call_method0("is_contiguous")?.extract::<bool>()?
    {
        return Err(PyRuntimeError::new_err(
            "invalid CNOT borrowed tensor storage",
        ));
    }
    let address: usize = tensor.call_method0("data_ptr")?.extract()?;
    let elements: u64 = tensor.call_method0("numel")?.extract()?;
    if elements != expected || (elements != 0 && address == 0) {
        return Err(PyRuntimeError::new_err(
            "invalid CNOT borrowed tensor extent",
        ));
    }
    Ok(DeviceView::from_raw(
        address as *mut c_void,
        elements,
        device,
    ))
}

fn check_matrices(py: Python<'_>, tensor: &Bound<'_, PyAny>, device: i32, n: i32) -> PyResult<u64> {
    let torch = py.import("torch")?;
    exact_tensor(&torch, tensor)?;
    let is_long = tensor.getattr("dtype")?.is(&torch.getattr("int64")?);
    let is_cuda = tensor.getattr("is_cuda")?.extract::<bool>()?;
    if !is_cuda {
        return Err(PyRuntimeError::new_err(
            "CNOT borrowed matrices require CUDA",
        ));
    }
    let same_device = tensor
        .getattr("device")?
        .getattr("index")?
        .extract::<i32>()?
        == device;
    let contiguous = tensor.call_method0("is_contiguous")?.is_truthy()?;
    let dimensions: i32 = tensor.call_method0("dim")?.extract()?;
    if !is_cuda || !same_device || !is_long || !contiguous || dimensions != 3 {
        return Err(PyRuntimeError::new_err(
            "invalid CNOT borrowed matrix storage",
        ));
    }
    let rows: i32 = tensor.call_method1("size", (1,))?.extract()?;
    let words: i32 = tensor.call_method1("size", (2,))?.extract()?;
    if rows != n || words != (n + 63) / 64 {
        return Err(PyRuntimeError::new_err(
            "invalid CNOT borrowed matrix geometry",
        ));
    }
    tensor.call_method1("size", (0,))?.extract()
}
