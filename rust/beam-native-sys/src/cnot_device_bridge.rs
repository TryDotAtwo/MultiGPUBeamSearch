//! Rank-local borrowed CUDA bridge for the CNOT full-matrix distinct stage.
//!
//! This opt-in transition keeps Torch tensor storage and the existing rank
//! collectives. The domain remains on the search's calling thread; no CUDA
//! address becomes a Rust reference or leaves the rank process.
use crate::cnot_domain::{
    Config, DeviceMoveType, DeviceUnseenWorkspace, DeviceView, Domain, DomainLibrary,
};
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
    parent_capacity: u64,
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
            parent_capacity: 0,
            closed_capacity: 0,
            child_capacity: 0,
        })
    }

    fn domain(&mut self, parents: u64, closed: u64, children: u64) -> PyResult<&mut Domain> {
        if self.domain.is_none()
            || parents > self.parent_capacity
            || closed > self.closed_capacity
            || children > self.child_capacity
        {
            let parent_capacity = parents.max(self.parent_capacity);
            let mut closed_capacity = closed.max(self.closed_capacity);
            let mut child_capacity = children.max(self.child_capacity);
            if closed_capacity
                .checked_add(child_capacity)
                .map_or(true, |sum| sum > i32::MAX as u64)
            {
                // Earlier batches may have had a different valid geometry.
                // Retaining both old maxima must not reject this batch.
                closed_capacity = closed;
                child_capacity = children;
            }
            let config = Config {
                device: self.device,
                n: self.n,
                world: self.world,
                parent_capacity,
                child_capacity,
                closed_capacity,
            };
            let next = self.library.create_device(config).map_err(native_error)?;
            if let Some(previous) = self.domain.take() {
                previous.close().map_err(native_error)?;
            }
            self.domain = Some(next);
            self.parent_capacity = parent_capacity;
            self.closed_capacity = closed_capacity;
            self.child_capacity = child_capacity;
        }
        self.domain
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("CNOT device domain is unavailable"))
    }

    /// Assign the owner of each complete packed CNOT matrix on this rank.
    pub fn owners<'py>(
        &mut self,
        py: Python<'py>,
        states: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let torch = py.import("torch")?;
        exact_tensor(&torch, states)?;
        let device = states.getattr("device")?;
        let states = states.call_method0("contiguous")?;
        let count = check_matrices(py, &states, self.device, self.n)?;
        let output = empty(py, &torch, &device, "int64", count)?;
        if count == 0 {
            return Ok(output);
        }
        let words = matrix_words(count, self.n)?;
        let state_view = tensor_view(&torch, &states, self.device, "int64", words)?;
        let owner_view = tensor_view(&torch, &output, self.device, "int64", count)?;
        let stream = current_stream(&torch, &device, self.device)?;
        let domain = self.domain(0, 0, count)?;
        // SAFETY: Torch owns separate, checked allocations and retains them
        // through the call. Input producers precede it on this current stream;
        // native validates device/overlap and synchronizes before return.
        unsafe { domain.owners_device(state_view, count, owner_view, stream as *mut c_void) }
            .map_err(native_error)?;
        Ok(output)
    }

    /// Expand checked, generated parent indices and physical moves on this rank.
    pub fn children<'py>(
        &mut self,
        py: Python<'py>,
        parents: &Bound<'py, PyAny>,
        indices: &Bound<'py, PyAny>,
        moves: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let torch = py.import("torch")?;
        for tensor in [parents, indices, moves] {
            exact_tensor(&torch, tensor)?;
        }
        let device = parents.getattr("device")?;
        let parents = parents.call_method0("contiguous")?;
        let indices = indices.call_method0("contiguous")?;
        let moves = moves.call_method0("contiguous")?;
        let parent_count = check_matrices(py, &parents, self.device, self.n)?;
        if indices.call_method0("dim")?.extract::<i32>()? != 1
            || moves.call_method0("dim")?.extract::<i32>()? != 1
        {
            return Err(PyRuntimeError::new_err("invalid CNOT descriptor geometry"));
        }
        let count = indices.call_method0("numel")?.extract::<u64>()?;
        let index_view = tensor_view(&torch, &indices, self.device, "int64", count)?;
        let move_view = tensor_view(&torch, &moves, self.device, "int64", count)?;
        let parent_words = matrix_words(parent_count, self.n)?;
        let child_words = matrix_words(count, self.n)?;
        let parent_view = tensor_view(&torch, &parents, self.device, "int64", parent_words)?;
        let child_count = i64::try_from(count)
            .map_err(|_| PyRuntimeError::new_err("CNOT child count overflow"))?;
        let output = empty(py, &torch, &device, "int64", child_words)?
            .call_method1("reshape", (child_count, self.n, (self.n + 63) / 64))?;
        if count == 0 {
            return Ok(output);
        }
        if parent_count == 0 {
            return Err(PyRuntimeError::new_err(
                "cannot expand an empty CNOT frontier",
            ));
        }
        // Validate descriptors before entering native code. A device-side
        // assert would poison this rank's CUDA context and prevent the
        // distributed caller from voting on a normal rank-local error.
        assert_descriptor_range(&torch, &indices, parent_count, "invalid CNOT parent index")?;
        let alphabet = u64::try_from(self.n * (self.n - 1))
            .map_err(|_| PyRuntimeError::new_err("invalid CNOT move alphabet"))?;
        assert_descriptor_range(&torch, &moves, alphabet, "invalid CNOT physical move")?;
        let child_view = tensor_view(&torch, &output, self.device, "int64", child_words)?;
        let stream = current_stream(&torch, &device, self.device)?;
        let domain = self.domain(parent_count, 0, count)?;
        // SAFETY: the live, contiguous, same-device tensors and separate
        // output were checked above. Descriptor validation and all producers
        // precede the synchronous native call on the current stream.
        unsafe {
            domain.expand_device(
                parent_view,
                parent_count,
                index_view,
                move_view,
                DeviceMoveType::I64,
                count,
                child_view,
                stream as *mut c_void,
            )
        }
        .map_err(native_error)?;
        Ok(output)
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
        let closed_words = matrix_words(closed_count, self.n)?;
        let child_words = matrix_words(child_count, self.n)?;
        let device_id = self.device;
        let domain = self.domain(0, closed_count, child_count)?;
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
        let stream = current_stream(&torch, &device, device_id)?;
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

fn matrix_words(count: u64, n: i32) -> PyResult<u64> {
    let n = u64::try_from(n).map_err(|_| PyRuntimeError::new_err("invalid CNOT qubit count"))?;
    let stride = n
        .checked_mul((n + 63) / 64)
        .ok_or_else(|| PyRuntimeError::new_err("CNOT matrix stride overflow"))?;
    count
        .checked_mul(stride)
        .ok_or_else(|| PyRuntimeError::new_err("CNOT matrix word count overflow"))
}

fn assert_descriptor_range(
    torch: &Bound<'_, PyModule>,
    values: &Bound<'_, PyAny>,
    upper: u64,
    message: &str,
) -> PyResult<()> {
    let upper = i64::try_from(upper)
        .map_err(|_| PyRuntimeError::new_err("CNOT descriptor bound overflow"))?;
    let lower_ok = values.call_method1("__ge__", (0,))?;
    let upper_ok = values.call_method1("__lt__", (upper,))?;
    let valid = torch
        .call_method1("logical_and", (lower_ok, upper_ok))?
        .call_method0("all")?;
    let valid = valid.call_method0("item")?.extract::<bool>()?;
    if !valid {
        return Err(PyRuntimeError::new_err(message.to_owned()));
    }
    Ok(())
}

fn current_stream(
    torch: &Bound<'_, PyModule>,
    device: &Bound<'_, PyAny>,
    device_id: i32,
) -> PyResult<usize> {
    let cuda = torch.getattr("cuda")?;
    let stream = cuda.call_method1("current_stream", (device,))?;
    if !stream.get_type().as_any().is(&cuda.getattr("Stream")?)
        || stream
            .getattr("device")?
            .getattr("index")?
            .extract::<i32>()?
            != device_id
    {
        return Err(PyRuntimeError::new_err(
            "CNOT domain requires a plain CUDA stream",
        ));
    }
    stream.getattr("cuda_stream")?.extract()
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

