//! Owner for the rank-local TheQuantumProject CNOT scorer ABI.
//!
//! PyTorch remains the owner of every CUDA allocation. This boundary owns the
//! dynamically loaded scorer and moves the existing fixed-size chunk loop out
//! of Python without changing its chunk sizes or call order.
use super::Error;
use libloading::Library;
#[cfg(feature = "python")]
use pyo3::{exceptions::PyValueError, prelude::*};
use std::{
    ffi::{c_char, c_void, CStr, CString},
    marker::PhantomData,
    path::Path,
    ptr::NonNull,
    rc::Rc,
};
type AbiVersion = unsafe extern "C" fn() -> u32;
type LastError = unsafe extern "C" fn() -> *const c_char;
type Open = unsafe extern "C" fn(*const c_char, u32, u32) -> *mut c_void;
type Metadata = unsafe extern "C" fn(*const c_void) -> *const c_char;
type Score = unsafe extern "C" fn(*const c_void, u32, i64, *const u8, *mut u8, *mut u8) -> i32;
type ScoreMany = unsafe extern "C" fn(
    *const c_void,
    u32,
    i64,
    i64,
    *const u8,
    usize,
    *mut u8,
    usize,
    *mut u8,
    usize,
) -> i32;
type Close = unsafe extern "C" fn(*mut c_void);

struct Api {
    _library: Library,
    error: LastError,
    open: Open,
    metadata: Metadata,
    score: Score,
    score_many: Option<ScoreMany>,
    close: Close,
}

impl Api {
    fn error(&self) -> Error {
        // SAFETY: the scorer ABI returns a thread-local, NUL-terminated error
        // string that remains valid until the next ABI call on this thread.
        let pointer = unsafe { (self.error)() };
        let detail = if pointer.is_null() {
            "CNOT scorer failed".to_owned()
        } else {
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        };
        Error(if detail.is_empty() {
            "CNOT scorer failed".into()
        } else {
            detail
        })
    }
}

/// A thread-bound scorer handle. Device addresses are accepted only for the
/// duration of an unsafe call and are never turned into Rust references.
pub struct TqpScorer {
    api: Api,
    handle: Option<NonNull<c_void>>,
    metadata: String,
    wires: u32,
    device: u32,
    native_batch: bool,
    _thread_bound: PhantomData<Rc<()>>,
}

impl TqpScorer {
    pub fn open(
        library_path: &Path,
        manifest_path: &Path,
        wires: u32,
        device: u32,
        native_batch: bool,
    ) -> Result<Self, Error> {
        if wires < 2 {
            return Err(Error("n must be a valid CNOT dimension".into()));
        }
        // SAFETY: the caller supplies the scorer shared library for this ABI;
        // function pointers remain valid while Api retains the Library.
        let library_path = library_path
            .canonicalize()
            .map_err(|error| Error(format!("resolve {}: {error}", library_path.display())))?;
        let library = unsafe { Library::new(&library_path) }
            .map_err(|error| Error(format!("load {}: {error}", library_path.display())))?;
        // SAFETY: each symbol is checked by name and the ABI version before
        // any context operation. The library is moved into Api below.
        unsafe {
            let version = *library
                .get::<AbiVersion>(b"mgbs_tqp_abi_version\0")
                .map_err(|error| Error(format!("CNOT scorer ABI unavailable: {error}")))?;
            let actual = version();
            if actual != 1 {
                return Err(Error(format!(
                    "unsupported CNOT scorer ABI: required=1 available={actual}"
                )));
            }
            macro_rules! symbol {
                ($name:literal, $ty:ty) => {
                    *library
                        .get::<$ty>(concat!($name, "\0").as_bytes())
                        .map_err(|error| {
                            Error(format!("CNOT scorer {} unavailable: {error}", $name))
                        })?
                };
            }
            let api = Api {
                error: symbol!("mgbs_tqp_error", LastError),
                open: symbol!("mgbs_tqp_open", Open),
                metadata: symbol!("mgbs_tqp_metadata", Metadata),
                score: symbol!("mgbs_tqp_score", Score),
                score_many: library
                    .get::<ScoreMany>(b"mgbs_tqp_score_many\0")
                    .ok()
                    .map(|symbol| *symbol),
                close: symbol!("mgbs_tqp_close", Close),
                _library: library,
            };
            if native_batch && api.score_many.is_none() {
                return Err(Error(
                    "the scorer library lacks the experimental batch scorer ABI".into(),
                ));
            }

            let manifest = manifest_path
                .canonicalize()
                .map_err(|error| Error(format!("resolve {}: {error}", manifest_path.display())))?;
            let manifest = manifest
                .to_str()
                .ok_or_else(|| Error("CNOT scorer manifest path is not UTF-8".into()))?;
            let manifest = CString::new(manifest)
                .map_err(|_| Error("CNOT scorer manifest path contains NUL".into()))?;
            let handle = NonNull::new((api.open)(manifest.as_ptr(), wires, device))
                .ok_or_else(|| api.error())?;
            let metadata_ptr = (api.metadata)(handle.as_ptr());
            if metadata_ptr.is_null() {
                (api.close)(handle.as_ptr());
                return Err(api.error());
            }
            let metadata = match CStr::from_ptr(metadata_ptr).to_str() {
                Ok(metadata) => metadata.to_owned(),
                Err(error) => {
                    (api.close)(handle.as_ptr());
                    return Err(Error(format!("CNOT scorer metadata is not UTF-8: {error}")));
                }
            };
            Ok(Self {
                api,
                handle: Some(handle),
                metadata,
                wires,
                device,
                native_batch,
                _thread_bound: PhantomData,
            })
        }
    }

    pub fn metadata(&self) -> &str {
        &self.metadata
    }

    pub fn supports_native_batch(&self) -> bool {
        self.api.score_many.is_some()
    }

    pub fn is_closed(&self) -> bool {
        self.handle.is_none()
    }

    pub fn close(&mut self) {
        if let Some(handle) = self.handle.take() {
            // SAFETY: this is the unique live handle returned by mgbs_tqp_open.
            unsafe { (self.api.close)(handle.as_ptr()) };
        }
    }

    /// Call the existing scorer over borrowed PyTorch CUDA buffers.
    ///
    /// # Safety
    /// The caller must keep all three allocations alive, contiguous, disjoint,
    /// on the scorer device, and valid for the declared byte extents until the
    /// queued scorer work has finished. The addresses must originate from
    /// PyTorch tensors on the calling rank; this function never dereferences
    /// them on the CPU.
    #[allow(clippy::too_many_arguments)]
    unsafe fn score_packet(
        &self,
        head: u32,
        batch: i64,
        batch_limit: i64,
        rows: usize,
        rows_bytes: usize,
        output: usize,
        output_bytes: usize,
        votes: usize,
        votes_bytes: usize,
    ) -> Result<(), Error> {
        if self.handle.is_none() {
            return Err(Error("CNOT scorer is closed".into()));
        }
        if head > 2 || batch < 0 || batch_limit < 1 {
            return Err(Error(
                "invalid scorer head, packet size or batch limit".into(),
            ));
        }
        if batch == 0 {
            return Ok(());
        }
        let count =
            usize::try_from(batch).map_err(|_| Error("CNOT packet size overflow".into()))?;
        let limit =
            usize::try_from(batch_limit).map_err(|_| Error("CNOT batch limit overflow".into()))?;
        let (row_stride, output_stride, votes_stride) = packet_strides(self.wires, head)?;
        validate_extent("rows", rows, rows_bytes, count, row_stride)?;
        validate_extent("output", output, output_bytes, count, output_stride)?;
        validate_extent("votes", votes, votes_bytes, count, votes_stride)?;
        let handle = self.handle.expect("checked live scorer handle");

        if self.native_batch {
            let score_many = self
                .api
                .score_many
                .expect("native batch ABI checked at open");
            // SAFETY: caller contract and byte/address extents were checked
            // above; these are borrowed device addresses, not host references.
            let code = unsafe {
                score_many(
                    handle.as_ptr(),
                    head,
                    batch,
                    batch_limit,
                    rows as *const u8,
                    rows_bytes,
                    output as *mut u8,
                    output_bytes,
                    votes as *mut u8,
                    votes_bytes,
                )
            };
            if code != 0 {
                return Err(self.api.error());
            }
            return Ok(());
        }

        for (start, size) in ScoreChunks::new(count, limit)? {
            let row_offset = start
                .checked_mul(row_stride)
                .ok_or_else(|| Error("CNOT row offset overflow".into()))?;
            let output_offset = start
                .checked_mul(output_stride)
                .ok_or_else(|| Error("CNOT output offset overflow".into()))?;
            let votes_offset = start
                .checked_mul(votes_stride)
                .ok_or_else(|| Error("CNOT status offset overflow".into()))?;
            let row_address = rows
                .checked_add(row_offset)
                .ok_or_else(|| Error("CNOT row pointer offset overflow".into()))?;
            let output_address = output
                .checked_add(output_offset)
                .ok_or_else(|| Error("CNOT output pointer offset overflow".into()))?;
            let votes_address = votes
                .checked_add(votes_offset)
                .ok_or_else(|| Error("CNOT status pointer offset overflow".into()))?;
            // SAFETY: the whole packet extents and every start offset were
            // checked; the caller keeps each borrowed CUDA allocation alive.
            let code = unsafe {
                (self.api.score)(
                    handle.as_ptr(),
                    head,
                    i64::try_from(size)
                        .map_err(|_| Error("CNOT subpacket size overflow".into()))?,
                    row_address as *const u8,
                    output_address as *mut u8,
                    votes_address as *mut u8,
                )
            };
            if code != 0 {
                return Err(self.api.error());
            }
        }
        Ok(())
    }

    /// Validate borrowed PyTorch tensors, then submit their addresses to the
    /// scorer ABI. This is the safe entry point used by beam-python; tensor
    /// owners remain alive for the full native call.
    #[cfg(feature = "python")]
    pub fn score_tensors(
        &self,
        py: Python<'_>,
        head: u32,
        batch_limit: i64,
        rows: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        votes: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if head > 2 {
            return Err(PyValueError::new_err("invalid scorer head"));
        }
        let torch = py.import("torch")?;
        let device = torch
            .getattr("device")?
            .call1((format!("cuda:{}", self.device),))?;
        let row_shape = tensor_shape(rows)?;
        if row_shape.len() != 3 {
            return Err(PyValueError::new_err(
                "rows must be rank-local CUDA int64 [batch, n, ceil(n/64)]",
            ));
        }
        let count = i64::try_from(row_shape[0])
            .map_err(|_| PyValueError::new_err("CNOT packet size overflow"))?;
        let n = usize::try_from(self.wires)
            .map_err(|_| PyValueError::new_err("CNOT dimension overflow"))?;
        let words = n
            .checked_add(63)
            .ok_or_else(|| PyValueError::new_err("CNOT row width overflow"))?
            / 64;
        if row_shape[1..] != [n, words] {
            return Err(PyValueError::new_err(
                "rows must be rank-local CUDA int64 [batch, n, ceil(n/64)]",
            ));
        }
        let output_shape = if head == 2 {
            vec![row_shape[0]]
        } else {
            let columns = n
                .checked_mul(
                    n.checked_sub(1)
                        .ok_or_else(|| PyValueError::new_err("invalid CNOT dimension"))?,
                )
                .ok_or_else(|| PyValueError::new_err("CNOT score width overflow"))?;
            vec![row_shape[0], columns]
        };
        let status_shape = vec![row_shape[0]];
        let rows_buffer = tensor_buffer(py, &torch, rows, "int64", &device, &row_shape)?;
        let output_buffer = tensor_buffer(py, &torch, output, "float32", &device, &output_shape)?;
        let votes_buffer = tensor_buffer(py, &torch, votes, "int64", &device, &status_shape)?;

        if ranges_overlap(
            rows_buffer.address,
            rows_buffer.bytes,
            output_buffer.address,
            output_buffer.bytes,
        ) || ranges_overlap(
            rows_buffer.address,
            rows_buffer.bytes,
            votes_buffer.address,
            votes_buffer.bytes,
        ) || ranges_overlap(
            output_buffer.address,
            output_buffer.bytes,
            votes_buffer.address,
            votes_buffer.bytes,
        ) {
            return Err(PyValueError::new_err(
                "CNOT scorer buffers must be disjoint PyTorch allocations",
            ));
        }
        // SAFETY: tensor_buffer checked CUDA device, dtype, exact shape,
        // contiguity, and each allocation's full byte range. The three Python
        // tensor owners stay borrowed until the ABI call has returned.
        unsafe {
            self.score_packet(
                head,
                count,
                batch_limit,
                rows_buffer.address,
                rows_buffer.bytes,
                output_buffer.address,
                output_buffer.bytes,
                votes_buffer.address,
                votes_buffer.bytes,
            )
        }
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }
}

impl Drop for TqpScorer {
    fn drop(&mut self) {
        self.close();
    }
}

fn packet_strides(wires: u32, head: u32) -> Result<(usize, usize, usize), Error> {
    if !(2..=u32::MAX).contains(&wires) || head > 2 {
        return Err(Error("invalid CNOT dimension or scorer head".into()));
    }
    let n = usize::try_from(wires).map_err(|_| Error("CNOT dimension overflow".into()))?;
    let words = n
        .checked_add(63)
        .ok_or_else(|| Error("CNOT row width overflow".into()))?
        / 64;
    let row_stride = n
        .checked_mul(words)
        .and_then(|value| value.checked_mul(8))
        .ok_or_else(|| Error("CNOT row byte stride overflow".into()))?;
    let columns = if head == 2 {
        1
    } else {
        n.checked_mul(
            n.checked_sub(1)
                .ok_or_else(|| Error("invalid CNOT dimension".into()))?,
        )
        .ok_or_else(|| Error("CNOT score width overflow".into()))?
    };
    let output_stride = columns
        .checked_mul(4)
        .ok_or_else(|| Error("CNOT output stride overflow".into()))?;
    Ok((row_stride, output_stride, 8))
}

fn validate_extent(
    name: &str,
    address: usize,
    available: usize,
    count: usize,
    stride: usize,
) -> Result<(), Error> {
    let required = count
        .checked_mul(stride)
        .ok_or_else(|| Error("CNOT packet byte size overflow".into()))?;
    if address == 0 {
        return Err(Error(format!("invalid CNOT {name} storage")));
    }
    if available < required {
        return Err(Error(format!(
            "CNOT {name} capacity exceeded: required={required} available={available}"
        )));
    }
    address
        .checked_add(required - 1)
        .ok_or_else(|| Error("CNOT device pointer offset overflow".into()))?;
    Ok(())
}

#[cfg(feature = "python")]
struct TensorBuffer {
    address: usize,
    bytes: usize,
}

#[cfg(feature = "python")]
fn tensor_shape(tensor: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    tensor.getattr("shape")?.extract()
}

#[cfg(feature = "python")]
fn tensor_buffer(
    py: Python<'_>,
    torch: &Bound<'_, PyAny>,
    tensor: &Bound<'_, PyAny>,
    dtype_name: &str,
    device: &Bound<'_, PyAny>,
    expected_shape: &[usize],
) -> PyResult<TensorBuffer> {
    if !torch
        .call_method1("is_tensor", (tensor,))?
        .extract::<bool>()?
    {
        return Err(PyValueError::new_err(
            "CNOT scorer storage must be a PyTorch tensor",
        ));
    }
    let operator = py.import("operator")?;
    let dtype_ok = operator
        .call_method1("eq", (tensor.getattr("dtype")?, torch.getattr(dtype_name)?))?
        .extract::<bool>()?;
    let device_ok = operator
        .call_method1("eq", (tensor.getattr("device")?, device))?
        .extract::<bool>()?;
    let shape = tensor_shape(tensor)?;
    let contiguous: bool = tensor.call_method0("is_contiguous")?.extract()?;
    if !dtype_ok || !device_ok || shape != expected_shape || !contiguous {
        return Err(PyValueError::new_err(
            "CNOT scorer storage has an invalid dtype, device, shape or stride",
        ));
    }
    let numel: usize = tensor.call_method0("numel")?.extract()?;
    let element_size: usize = tensor.call_method0("element_size")?.extract()?;
    let bytes = numel
        .checked_mul(element_size)
        .ok_or_else(|| PyValueError::new_err("CNOT tensor byte size overflow"))?;
    let address: usize = tensor.call_method0("data_ptr")?.extract()?;
    if address == 0 || bytes == 0 {
        return Err(PyValueError::new_err("CNOT scorer storage is empty"));
    }
    address
        .checked_add(bytes - 1)
        .ok_or_else(|| PyValueError::new_err("CNOT device pointer offset overflow"))?;
    Ok(TensorBuffer { address, bytes })
}

#[cfg(feature = "python")]
fn ranges_overlap(first: usize, first_len: usize, second: usize, second_len: usize) -> bool {
    if first <= second {
        second - first < first_len
    } else {
        first - second < second_len
    }
}

struct ScoreChunks {
    count: usize,
    limit: usize,
    next: usize,
}

impl ScoreChunks {
    fn new(count: usize, limit: usize) -> Result<Self, Error> {
        if limit == 0 {
            return Err(Error("CNOT scorer batch limit must be positive".into()));
        }
        Ok(Self {
            count,
            limit,
            next: 0,
        })
    }
}

impl Iterator for ScoreChunks {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.count {
            return None;
        }
        let start = self.next;
        let size = self.limit.min(self.count - start);
        self.next += size;
        Some((start, size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunking_preserves_the_existing_inference_batch_partition() {
        let chunks: Vec<_> = ScoreChunks::new(10, 4).unwrap().collect();
        assert_eq!(chunks, [(0, 4), (4, 4), (8, 2)]);
    }

    #[test]
    fn empty_packet_produces_no_scorer_call() {
        assert_eq!(ScoreChunks::new(0, 4).unwrap().count(), 0);
    }

    #[test]
    fn native_packet_strides_match_the_packed_cnot_scorer_layout() {
        assert_eq!(packet_strides(4, 0).unwrap(), (32, 48, 8));
        assert_eq!(packet_strides(4, 1).unwrap(), (32, 48, 8));
        assert_eq!(packet_strides(4, 2).unwrap(), (32, 4, 8));
    }

    fn metadata_json(feature_count: usize, id_count: usize, capacity: i64) -> String {
        let feature_ids = (0..id_count)
            .map(|index| format!("feature-{index}"))
            .collect::<Vec<_>>();
        serde_json::json!({
            "feature_model_inputs": feature_count,
            "feature_ids": feature_ids,
            "max_packet_rows": capacity,
        })
        .to_string()
    }

    #[test]
    fn scorer_metadata_preserves_feature_ids_and_caps_the_requested_batch() {
        let raw = metadata_json(102, 102, 17);
        let metadata = CnotScorerMetadata::parse(&raw).unwrap();
        assert_eq!(metadata.feature_ids().len(), 102);
        assert_eq!(metadata.feature_ids()[0], "feature-0");
        assert_eq!(metadata.feature_ids()[101], "feature-101");
        assert_eq!(metadata.effective_batch(64).unwrap(), 17);
        assert_eq!(metadata.effective_batch(4).unwrap(), 4);
    }

    #[test]
    fn scorer_metadata_rejects_a_partial_registered_feature_contract() {
        for raw in [metadata_json(101, 102, 17), metadata_json(102, 101, 17)] {
            let error = CnotScorerMetadata::parse(&raw).unwrap_err();
            assert_eq!(
                error.to_string(),
                "expected all 102 registered field inputs"
            );
        }
    }

    #[test]
    fn scorer_metadata_rejects_malformed_json() {
        let error = CnotScorerMetadata::parse("{").unwrap_err();
        assert!(error
            .to_string()
            .starts_with("CNOT scorer metadata is not valid JSON:"));
    }

    #[test]
    fn scorer_metadata_rejects_a_feature_policy_that_admits_no_rows() {
        let metadata = CnotScorerMetadata::parse(&metadata_json(102, 102, 0)).unwrap();
        let error = metadata.effective_batch(4).unwrap_err();
        assert_eq!(
            error.to_string(),
            "the feature policy cannot admit a single row"
        );
    }

    #[test]
    fn score_stride_overflow_is_rejected_before_any_abi_call() {
        assert!(packet_strides(u32::MAX, 0)
            .unwrap_err()
            .to_string()
            .contains("output stride overflow"));
    }
}
