//! Owned, thread-bound handle for the existing versioned CNOT Stream 5 ABI.
//!
//! This module does not load CUDA until explicitly requested, and does not
//! expose device pointers to safe callers. Its native owner retains the
//! initialization device and restores the caller's current device after
//! cleanup. The owner remains bound to its creating thread. The native library
//! stays loaded until every communicator is closed or dropped.
use super::{Error, Result};
use libloading::Library;
use std::{
    ffi::{c_char, c_int, c_void, CStr},
    path::Path,
    ptr::NonNull,
    rc::Rc,
};

const ABI: c_int = 1;
const OWNED_ABI: c_int = 1;
const MAX_ID_BYTES: usize = 4096;
const MAX_WORLD: u32 = 65536;

type Version = unsafe extern "C" fn() -> c_int;
type IdSize = unsafe extern "C" fn() -> c_int;
type CreateId = unsafe extern "C" fn(*mut c_void) -> c_int;
type Init = unsafe extern "C" fn(*mut *mut c_void, *const c_void, c_int, c_int) -> c_int;
type Close = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type LastError = unsafe extern "C" fn() -> *const c_char;
type Device = unsafe extern "C" fn(*mut c_void, *mut c_int) -> c_int;
type Counts = unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, *mut c_void) -> c_int;
type Rows = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *const c_void,
    *mut c_void,
    *const u32,
    *const u32,
    u32,
) -> c_int;
type AllReduceI64 = unsafe extern "C" fn(*mut c_void, *mut c_void, u64, *mut c_void) -> c_int;
type AllGatherI64 =
    unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void, u64, *mut c_void) -> c_int;
type BroadcastI64 =
    unsafe extern "C" fn(*mut c_void, *mut c_void, u64, c_int, *mut c_void) -> c_int;

struct Exchange {
    device: Device,
    counts: Counts,
    rows: Rows,
}

struct Collectives {
    all_reduce_i64: AllReduceI64,
    all_gather_i64: AllGatherI64,
    broadcast_i64: BroadcastI64,
}

struct Bindings {
    _library: Library,
    id_size: usize,
    create_id: CreateId,
    init: Init,
    close: Close,
    last_error: LastError,
    exchange: Option<Exchange>,
    collectives: Option<Collectives>,
}

impl Bindings {
    fn error(&self, operation: &str, code: c_int) -> Error {
        // SAFETY: the versioned native ABI returns a thread-local, NUL-terminated
        // error string, valid until the next call on this same thread.
        let ptr = unsafe { (self.last_error)() };
        let detail = if ptr.is_null() {
            "unknown native error".to_string()
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        };
        Error(format!(
            "CNOT Stream 5 {operation} failed ({code}): {detail}"
        ))
    }
}

/// A dynamically loaded ABI. Rc makes the library and its contexts thread-bound.
#[derive(Clone)]
pub struct Stream5Library(Rc<Bindings>);

impl Stream5Library {
    pub fn load(path: &Path) -> Result<Self> {
        // SAFETY: the caller selects a native library implementing this ABI.
        // The owned Library remains alive in every Rc<Bindings> handle.
        unsafe {
            let library = Library::new(path)
                .map_err(|error| Error(format!("load {}: {error}", path.display())))?;
            let version = *library
                .get::<Version>(b"cnot_s5_abi\0")
                .map_err(|error| Error(error.to_string()))?;
            let actual = version();
            if actual != ABI {
                return Err(Error(format!(
                    "CNOT Stream 5 ABI mismatch: required={ABI} available={actual}"
                )));
            }
            let owned_version = *library
                .get::<Version>(b"cnot_s5_owned_abi\0")
                .map_err(|error| Error(format!("CNOT Stream 5 owned ABI unavailable: {error}")))?;
            let owned_actual = owned_version();
            if owned_actual != OWNED_ABI {
                return Err(Error(format!(
                    "CNOT Stream 5 owned ABI mismatch: required={OWNED_ABI} available={owned_actual}"
                )));
            }
            macro_rules! symbol {
                ($name:literal, $ty:ty) => {
                    *library
                        .get::<$ty>(concat!($name, "\0").as_bytes())
                        .map_err(|error| Error(error.to_string()))?
                };
            }
            let size = symbol!("cnot_s5_id_bytes", IdSize)();
            if size <= 0 || size as usize > MAX_ID_BYTES {
                return Err(Error(format!(
                    "invalid CNOT Stream 5 identity size: {size}"
                )));
            }
            let exchange = match library.get::<Version>(b"cnot_s5_owned_exchange_abi\0") {
                Ok(version) => {
                    let actual = version();
                    if actual != 1 {
                        return Err(Error(format!(
                            "CNOT Stream 5 exchange ABI mismatch: required=1 available={actual}"
                        )));
                    }
                    Some(Exchange {
                        device: symbol!("cnot_s5_owned_device", Device),
                        counts: symbol!("cnot_s5_owned_counts", Counts),
                        rows: symbol!("cnot_s5_owned_rows", Rows),
                    })
                }
                Err(_) => None,
            };
            let collectives = match library.get::<Version>(b"cnot_s5_owned_collectives_abi\0") {
                Ok(version) => {
                    let actual = version();
                    if actual != 1 {
                        return Err(Error(format!(
                            "CNOT Stream 5 collectives ABI mismatch: required=1 available={actual}"
                        )));
                    }
                    Some(Collectives {
                        all_reduce_i64: symbol!("cnot_s5_owned_allreduce_i64", AllReduceI64),
                        all_gather_i64: symbol!("cnot_s5_owned_allgather_i64", AllGatherI64),
                        broadcast_i64: symbol!("cnot_s5_owned_broadcast_i64", BroadcastI64),
                    })
                }
                Err(_) => None,
            };
            Ok(Self(Rc::new(Bindings {
                id_size: size as usize,
                create_id: symbol!("cnot_s5_id", CreateId),
                init: symbol!("cnot_s5_owned_init", Init),
                close: symbol!("cnot_s5_owned_close", Close),
                last_error: symbol!("cnot_s5_error", LastError),
                exchange,
                collectives,
                _library: library,
            })))
        }
    }

    pub fn supports_exchange(&self) -> bool {
        self.0.exchange.is_some()
    }

    pub fn supports_collectives(&self) -> bool {
        self.0.collectives.is_some()
    }

    pub fn id_size(&self) -> usize {
        self.0.id_size
    }

    pub fn unique_id(&self) -> Result<Vec<u8>> {
        let mut bytes = vec![0; self.0.id_size];
        // SAFETY: the checked ABI writes exactly id_size bytes into this buffer.
        let code = unsafe { (self.0.create_id)(bytes.as_mut_ptr().cast()) };
        if code != 0 {
            return Err(self.0.error("create identity", code));
        }
        Ok(bytes)
    }

    pub fn init(&self, id: &[u8], rank: u32, world: u32) -> Result<Stream5Communicator> {
        if id.len() != self.0.id_size {
            return Err(Error(format!(
                "CNOT Stream 5 identity size: required={} available={}",
                self.0.id_size,
                id.len()
            )));
        }
        if world == 0 || world > MAX_WORLD || rank >= world {
            return Err(Error(format!(
                "invalid CNOT Stream 5 rank {rank} of {world}"
            )));
        }
        let mut raw = std::ptr::null_mut();
        // SAFETY: the ID has the native ABI's exact size; raw is a valid out
        // pointer. Both remain alive for this synchronous native call.
        let code =
            unsafe { (self.0.init)(&mut raw, id.as_ptr().cast(), rank as c_int, world as c_int) };
        if code != 0 {
            return Err(self.0.error("initialize communicator", code));
        }
        let ptr = NonNull::new(raw)
            .ok_or_else(|| Error("CNOT Stream 5 returned a null communicator".into()))?;
        Ok(Stream5Communicator {
            bindings: self.0.clone(),
            ptr: Some(ptr),
            rank,
            world,
        })
    }
}

/// Owns one NCCL communicator; never send it to another thread or process.
pub struct Stream5Communicator {
    bindings: Rc<Bindings>,
    ptr: Option<NonNull<c_void>>,
    rank: u32,
    world: u32,
}

impl Stream5Communicator {
    pub fn is_closed(&self) -> bool {
        self.ptr.is_none()
    }

    pub fn close(&mut self, abort: bool) -> Result<()> {
        let Some(ptr) = self.ptr.take() else {
            return Ok(());
        };
        // SAFETY: this communicator was returned by init on this thread and is
        // consumed exactly once. The library outlives the call via bindings.
        let code = unsafe { (self.bindings.close)(ptr.as_ptr(), c_int::from(abort)) };
        if code != 0 {
            return Err(self.bindings.error("close communicator", code));
        }
        Ok(())
    }
}

impl Drop for Stream5Communicator {
    fn drop(&mut self) {
        // Explicit close remains preferred; abandonment aborts the owned rank.
        let _ = self.close(true);
    }
}

#[cfg(feature = "python")]
mod torch {
    use super::*;
    use pyo3::{
        exceptions::{PyRuntimeError, PyValueError},
        prelude::*,
    };

    fn invalid(message: impl Into<String>) -> PyErr {
        PyValueError::new_err(message.into())
    }

    // Only exact Torch tensors are accepted; a subclass can override data_ptr.
    // No Rust reference is constructed from the returned CUDA address.
    fn shape(
        tensor: &Bound<'_, PyAny>,
        dtype: &str,
        device: i32,
        dims: usize,
    ) -> PyResult<Vec<u64>> {
        let torch = tensor.py().import("torch")?;
        if !tensor.get_type().as_any().is(&torch.getattr("Tensor")?) {
            return Err(invalid("Stream 5 requires a plain torch.Tensor"));
        }
        if !tensor.getattr("is_cuda")?.extract::<bool>()? {
            return Err(invalid("Stream 5 requires CUDA tensors"));
        }
        if !tensor.getattr("dtype")?.is(&torch.getattr(dtype)?) {
            return Err(invalid(format!("Stream 5 tensor requires dtype {dtype}")));
        }
        if tensor
            .getattr("device")?
            .getattr("index")?
            .extract::<i32>()?
            != device
        {
            return Err(invalid("Stream 5 tensor is on a different CUDA device"));
        }
        if !tensor.call_method0("is_contiguous")?.extract::<bool>()? {
            return Err(invalid("Stream 5 tensor must be contiguous"));
        }
        let result: Vec<u64> = tensor.getattr("shape")?.extract()?;
        if result.len() != dims {
            return Err(invalid(format!(
                "Stream 5 tensor requires {dims} dimensions"
            )));
        }
        Ok(result)
    }

    fn current_stream<'py>(py: Python<'py>, device: i32) -> PyResult<Bound<'py, PyAny>> {
        let cuda = py.import("torch")?.getattr("cuda")?;
        let current: i32 = cuda.call_method0("current_device")?.extract()?;
        if current != device {
            return Err(invalid(
                "current CUDA device differs from Stream 5 communicator device",
            ));
        }
        cuda.call_method1("current_stream", (device,))
    }

    fn address(tensor: &Bound<'_, PyAny>, stream: &Bound<'_, PyAny>) -> PyResult<usize> {
        // record_stream prevents PyTorch's caching allocator from reusing the
        // storage before this asynchronous NCCL/CUDA stream has finished.
        tensor.call_method1("record_stream", (stream,))?;
        let pointer: usize = tensor.call_method0("data_ptr")?.extract()?;
        if pointer == 0 {
            return Err(invalid("null Stream 5 CUDA tensor address"));
        }
        Ok(pointer)
    }

    impl Stream5Communicator {
        fn exchange_bindings(&self) -> PyResult<(&Exchange, NonNull<c_void>, i32)> {
            let ptr = self
                .ptr
                .ok_or_else(|| PyRuntimeError::new_err("CNOT Stream 5 communicator is closed"))?;
            let exchange = self.bindings.exchange.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("CNOT Stream 5 exchange ABI is unavailable")
            })?;
            let mut device = -1;
            // SAFETY: ptr is the live owned handle, device is a valid out pointer,
            // and the loaded library remains alive via self.bindings.
            let code = unsafe { (exchange.device)(ptr.as_ptr(), &mut device) };
            if code != 0 {
                return Err(PyRuntimeError::new_err(
                    self.bindings.error("get device", code).to_string(),
                ));
            }
            Ok((exchange, ptr, device))
        }

        fn collective_bindings(&self) -> PyResult<(&Collectives, NonNull<c_void>, i32)> {
            let (_, ptr, device) = self.exchange_bindings()?;
            let collectives = self.bindings.collectives.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("CNOT Stream 5 collectives ABI is unavailable")
            })?;
            Ok((collectives, ptr, device))
        }

        fn control_tensor(tensor: &Bound<'_, PyAny>, device: i32) -> PyResult<(u64, bool)> {
            let torch = tensor.py().import("torch")?;
            if !tensor.get_type().as_any().is(&torch.getattr("Tensor")?) {
                return Err(invalid("Stream 5 requires a plain torch.Tensor"));
            }
            if !tensor.getattr("is_cuda")?.extract::<bool>()? {
                return Err(invalid("Stream 5 requires CUDA tensors"));
            }
            if !tensor.getattr("dtype")?.is(&torch.getattr("int64")?) {
                return Err(invalid("Stream 5 tensor requires dtype int64"));
            }
            if !tensor.getattr("layout")?.is(&torch.getattr("strided")?) {
                return Err(invalid("Stream 5 tensor requires strided storage"));
            }
            if tensor
                .getattr("device")?
                .getattr("index")?
                .extract::<i32>()?
                != device
            {
                return Err(invalid("Stream 5 tensor is on a different CUDA device"));
            }
            let count = tensor.call_method0("numel")?.extract::<u64>()?;
            let contiguous = tensor.call_method0("is_contiguous")?.extract::<bool>()?;
            Ok((count, contiguous))
        }

        pub fn validate_tensors(
            &self,
            send: &Bound<'_, PyAny>,
            counts: &Bound<'_, PyAny>,
            world: &Bound<'_, PyAny>,
        ) -> PyResult<()> {
            // Reject CPU inputs before trying any native handle or collective.
            if !send.getattr("is_cuda")?.extract::<bool>()?
                || !counts.getattr("is_cuda")?.extract::<bool>()?
            {
                return Err(invalid("Stream 5 requires CUDA tensors"));
            }
            let (_, _, device) = self.exchange_bindings()?;
            if world.getattr("rank")?.extract::<u32>()? != self.rank
                || world.getattr("size")?.extract::<u32>()? != self.world
            {
                return Err(invalid(
                    "Stream 5 communicator and world rank geometry differ",
                ));
            }
            if world
                .getattr("device")?
                .getattr("index")?
                .extract::<i32>()?
                != device
            {
                return Err(invalid("Stream 5 world is on a different CUDA device"));
            }
            let send_shape = shape(send, "int64", device, 2)?;
            let count_shape = shape(counts, "int64", device, 1)?;
            if send_shape[1] == 0
                || send_shape[1] > u64::from(u32::MAX)
                || count_shape[0] != u64::from(self.world)
            {
                return Err(invalid(
                    "invalid Stream 5 record width or peer-count length",
                ));
            }
            let _ = current_stream(send.py(), device)?;
            Ok(())
        }

        pub fn exchange_counts_tensors(
            &self,
            send: &Bound<'_, PyAny>,
            receive: &Bound<'_, PyAny>,
        ) -> PyResult<()> {
            let (exchange, ptr, device) = self.exchange_bindings()?;
            if shape(send, "int32", device, 1)? != [u64::from(self.world)]
                || shape(receive, "int32", device, 1)? != [u64::from(self.world)]
            {
                return Err(invalid("invalid Stream 5 count-buffer length"));
            }
            let stream = current_stream(send.py(), device)?;
            let src = address(send, &stream)?;
            let dst = address(receive, &stream)?;
            if src == dst {
                return Err(invalid("Stream 5 count buffers overlap"));
            }
            let raw_stream: usize = stream.getattr("cuda_stream")?.extract()?;
            // SAFETY: PyTorch owns both checked CUDA tensors and records this
            // stream before the asynchronous call. The pointers remain device
            // addresses; the native ABI receives only a borrowed view.
            let code = unsafe {
                (exchange.counts)(
                    ptr.as_ptr(),
                    raw_stream as *mut c_void,
                    src as *const c_void,
                    dst as *mut c_void,
                )
            };
            if code != 0 {
                return Err(PyRuntimeError::new_err(
                    self.bindings.error("exchange counts", code).to_string(),
                ));
            }
            Ok(())
        }

        pub fn exchange_row_tensors(
            &self,
            send: &Bound<'_, PyAny>,
            receive: &Bound<'_, PyAny>,
            sends: &[u32],
            receives: &[u32],
        ) -> PyResult<()> {
            let (exchange, ptr, device) = self.exchange_bindings()?;
            if sends.len() != self.world as usize || receives.len() != self.world as usize {
                return Err(invalid("invalid Stream 5 peer-count length"));
            }
            let send_shape = shape(send, "int64", device, 2)?;
            let recv_shape = shape(receive, "int64", device, 2)?;
            let sent: u64 = sends.iter().map(|&x| u64::from(x)).sum();
            let received: u64 = receives.iter().map(|&x| u64::from(x)).sum();
            if send_shape[1] == 0
                || send_shape[1] > u64::from(u32::MAX)
                || send_shape[1] != recv_shape[1]
                || send_shape[0] < sent.max(1)
                || recv_shape[0] < received.max(1)
                || sent > u64::from(u32::MAX)
                || received > u64::from(u32::MAX)
            {
                return Err(invalid("Stream 5 payload capacity or stride is invalid"));
            }
            let stream = current_stream(send.py(), device)?;
            let src = address(send, &stream)?;
            let dst = address(receive, &stream)?;
            if src == dst {
                return Err(invalid("Stream 5 payload buffers overlap"));
            }
            let raw_stream: usize = stream.getattr("cuda_stream")?.extract()?;
            // SAFETY: the checked PyTorch CUDA tensors are recorded on the
            // stream; the host count slices outlive the synchronous ABI call.
            let code = unsafe {
                (exchange.rows)(
                    ptr.as_ptr(),
                    raw_stream as *mut c_void,
                    src as *const c_void,
                    dst as *mut c_void,
                    sends.as_ptr(),
                    receives.as_ptr(),
                    send_shape[1] as u32,
                )
            };
            if code != 0 {
                return Err(PyRuntimeError::new_err(
                    self.bindings.error("exchange rows", code).to_string(),
                ));
            }
            Ok(())
        }

        pub fn all_reduce_i64(&self, tensor: &Bound<'_, PyAny>) -> PyResult<()> {
            let (collectives, ptr, device) = self.collective_bindings()?;
            let (count, contiguous) = Self::control_tensor(tensor, device)?;
            if count == 0 {
                return Ok(());
            }
            let work = if contiguous {
                tensor.clone()
            } else {
                tensor.call_method0("contiguous")?
            };
            let stream = current_stream(tensor.py(), device)?;
            if !contiguous {
                tensor.call_method1("record_stream", (&stream,))?;
            }
            let data = address(&work, &stream)?;
            let raw_stream: usize = stream.getattr("cuda_stream")?.extract()?;
            // SAFETY: work is a checked, stream-recorded PyTorch CUDA tensor;
            // NCCL performs an in-place reduction on this borrowed allocation.
            let code = unsafe {
                (collectives.all_reduce_i64)(
                    ptr.as_ptr(),
                    data as *mut c_void,
                    count,
                    raw_stream as *mut c_void,
                )
            };
            if code != 0 {
                return Err(PyRuntimeError::new_err(
                    self.bindings.error("all-reduce int64", code).to_string(),
                ));
            }
            if !contiguous {
                tensor.call_method1("copy_", (&work,))?;
            }
            Ok(())
        }

        pub fn all_gather_i64<'py>(
            &self,
            tensor: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let (_, _, device) = self.collective_bindings()?;
            let (count, contiguous) = Self::control_tensor(tensor, device)?;
            count
                .checked_mul(u64::from(self.world))
                .ok_or_else(|| invalid("Stream 5 fixed gather size overflow"))?;
            let mut output_shape = vec![u64::from(self.world)];
            output_shape.extend(tensor.getattr("shape")?.extract::<Vec<u64>>()?);
            let output = tensor.call_method1("new_empty", (output_shape,))?;
            let work = if contiguous {
                tensor.clone()
            } else {
                tensor.call_method0("contiguous")?
            };
            self.all_gather_i64_into(tensor, &work, &output)
        }

        /// Validate every rank-local fixed-gather buffer before entering NCCL.
        pub fn validate_all_gather_i64_into(
            &self,
            tensor: &Bound<'_, PyAny>,
            work: &Bound<'_, PyAny>,
            output: &Bound<'_, PyAny>,
        ) -> PyResult<()> {
            let (_, _, device) = self.collective_bindings()?;
            let (count, _) = Self::control_tensor(tensor, device)?;
            let (work_count, work_contiguous) = Self::control_tensor(work, device)?;
            let (output_count, output_contiguous) = Self::control_tensor(output, device)?;
            if !work_contiguous || work_count != count {
                return Err(invalid("Stream 5 fixed gather input buffer is invalid"));
            }
            if !output_contiguous {
                return Err(invalid("Stream 5 fixed gather output must be contiguous"));
            }

            let input_shape: Vec<u64> = tensor.getattr("shape")?.extract()?;
            let work_shape: Vec<u64> = work.getattr("shape")?.extract()?;
            if work_shape != input_shape {
                return Err(invalid("Stream 5 fixed gather input shape changed"));
            }
            let mut expected_shape = vec![u64::from(self.world)];
            expected_shape.extend(input_shape);
            let output_shape: Vec<u64> = output.getattr("shape")?.extract()?;
            if output_shape != expected_shape {
                return Err(invalid("Stream 5 fixed gather output shape is invalid"));
            }
            let received = count
                .checked_mul(u64::from(self.world))
                .ok_or_else(|| invalid("Stream 5 fixed gather size overflow"))?;
            if output_count != received {
                return Err(invalid("Stream 5 fixed gather output size is invalid"));
            }
            if count == 0 {
                return Ok(());
            }

            let stream = current_stream(tensor.py(), device)?;
            let send = address(work, &stream)?;
            let receive = address(output, &stream)?;
            if send == receive {
                return Err(invalid("Stream 5 fixed gather buffers overlap"));
            }
            let _: usize = stream.getattr("cuda_stream")?.extract()?;
            Ok(())
        }

        pub fn all_gather_i64_into<'py>(
            &self,
            tensor: &Bound<'py, PyAny>,
            work: &Bound<'py, PyAny>,
            output: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            self.validate_all_gather_i64_into(tensor, work, output)?;
            let (collectives, ptr, device) = self.collective_bindings()?;
            let (count, _) = Self::control_tensor(tensor, device)?;
            if count == 0 {
                return Ok(output.clone());
            }

            let stream = current_stream(tensor.py(), device)?;
            let send = address(work, &stream)?;
            let receive = address(output, &stream)?;
            let raw_stream: usize = stream.getattr("cuda_stream")?.extract()?;
            // SAFETY: both buffers are validated, contiguous, stream-recorded
            // tensors with output storage sized for world * count int64 elements.
            let code = unsafe {
                (collectives.all_gather_i64)(
                    ptr.as_ptr(),
                    send as *const c_void,
                    receive as *mut c_void,
                    count,
                    raw_stream as *mut c_void,
                )
            };
            if code != 0 {
                return Err(PyRuntimeError::new_err(
                    self.bindings.error("all-gather int64", code).to_string(),
                ));
            }
            Ok(output.clone())
        }


        pub fn broadcast_i64(&self, tensor: &Bound<'_, PyAny>, source: u32) -> PyResult<()> {
            let (collectives, ptr, device) = self.collective_bindings()?;
            if source >= self.world {
                return Err(invalid("Stream 5 broadcast source rank is invalid"));
            }
            let (count, contiguous) = Self::control_tensor(tensor, device)?;
            if count == 0 {
                return Ok(());
            }
            let work = if contiguous {
                tensor.clone()
            } else {
                tensor.call_method0("contiguous")?
            };
            let stream = current_stream(tensor.py(), device)?;
            if !contiguous {
                tensor.call_method1("record_stream", (&stream,))?;
            }
            let data = address(&work, &stream)?;
            let raw_stream: usize = stream.getattr("cuda_stream")?.extract()?;
            // SAFETY: work is a checked, stream-recorded CUDA tensor; NCCL's
            // in-place broadcast updates the same borrowed allocation on each rank.
            let code = unsafe {
                (collectives.broadcast_i64)(
                    ptr.as_ptr(),
                    data as *mut c_void,
                    count,
                    source as c_int,
                    raw_stream as *mut c_void,
                )
            };
            if code != 0 {
                return Err(PyRuntimeError::new_err(
                    self.bindings.error("broadcast int64", code).to_string(),
                ));
            }
            if !contiguous {
                tensor.call_method1("copy_", (&work,))?;
            }
            Ok(())
        }
    }
}
